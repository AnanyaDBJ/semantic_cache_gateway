import hashlib
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text

from config import Settings
from db.connection import LakebasePool
from cache.embedding import EmbeddingClient
from cache.models import CacheHit, ChatCompletionResponse

logger = logging.getLogger(__name__)


def build_context_string(messages: list[dict], max_turns: int = 5) -> str:
    """
    Build normalized context string for embedding.
    Includes system prompt + last N conversation turns.
    Same question in different contexts produces different embeddings.
    """
    parts = []

    system_msgs = [m for m in messages if m.get("role") == "system"]
    if system_msgs:
        parts.append(f"[SYSTEM] {system_msgs[-1]['content'].strip()}")

    conversation = [m for m in messages if m.get("role") != "system"]
    recent = conversation[-(max_turns * 2):]

    for msg in recent:
        role = msg["role"].upper()
        content = msg["content"].strip()
        parts.append(f"[{role}] {content}")

    return "\n".join(parts)


class SemanticCacheService:
    """
    Two-tier semantic cache: SHA-256 exact match → pgvector HNSW search.
    All operations are async and designed for production use.
    """

    def __init__(self, pool: LakebasePool, embedding_client: EmbeddingClient, settings: Settings):
        self.pool = pool
        self.embedding_client = embedding_client
        self.settings = settings
        self._eviction_task: Optional[asyncio.Task] = None

    def start_eviction_loop(self) -> None:
        self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.eviction_interval_seconds)
            try:
                async with self.pool.session() as session:
                    result = await session.execute(
                        text(
                            "DELETE FROM cache_entries "
                            "WHERE expires_at <= NOW()"
                        )
                    )
                    await session.commit()
                    if result.rowcount and result.rowcount > 0:
                        logger.info(f"Evicted {result.rowcount} expired cache entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Eviction loop error: {e}")

    async def lookup(
        self,
        context_string: str,
        owner_id: str,
        model_name: str,
    ) -> tuple[Optional[CacheHit], Optional[list[float]]]:
        """
        Two-tier cache lookup.
        Returns (hit_or_none, embedding_or_none).
        Embedding is returned so it can be reused for store on miss.
        """
        prompt_hash = hashlib.sha256(context_string.encode()).hexdigest()
        start = time.perf_counter()

        # Tier 1: Exact hash match (<5ms)
        exact = await self._exact_lookup(prompt_hash, owner_id, model_name)
        if exact:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(f"Cache HIT (exact) owner={owner_id} latency={elapsed_ms:.1f}ms")
            await self._log_event("hit", owner_id, model_name, 1.0, elapsed_ms, 0, exact.total_tokens)
            return exact, None

        # Tier 2: Semantic HNSW search
        embed_start = time.perf_counter()
        embedding = await self.embedding_client.embed(context_string)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        semantic = await self._semantic_lookup(embedding, owner_id, model_name)

        if semantic and semantic.similarity >= self.settings.similarity_threshold:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"Cache HIT (semantic) owner={owner_id} "
                f"sim={semantic.similarity:.3f} latency={elapsed_ms:.1f}ms"
            )
            await self._log_event(
                "hit", owner_id, model_name, semantic.similarity,
                elapsed_ms, embed_ms, semantic.total_tokens
            )
            return semantic, embedding

        # Miss
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"Cache MISS owner={owner_id} latency={elapsed_ms:.1f}ms")
        await self._log_event("miss", owner_id, model_name, None, elapsed_ms, embed_ms, 0)
        return None, embedding

    async def _exact_lookup(
        self, prompt_hash: str, owner_id: str, model_name: str
    ) -> Optional[CacheHit]:
        async with self.pool.session() as session:
            result = await session.execute(
                text("""
                    SELECT id, response_text, response_metadata,
                           prompt_tokens, completion_tokens, total_tokens
                    FROM cache_entries
                    WHERE prompt_hash = :hash
                      AND owner_id = :owner
                      AND model_name = :model
                      AND expires_at > NOW()
                    LIMIT 1
                """),
                {"hash": prompt_hash, "owner": owner_id, "model": model_name},
            )
            row = result.fetchone()
            if not row:
                return None

            # Update hit count asynchronously
            await session.execute(
                text("""
                    UPDATE cache_entries
                    SET hit_count = hit_count + 1, last_accessed_at = NOW()
                    WHERE id = :id
                """),
                {"id": str(row[0])},
            )
            await session.commit()

            metadata = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
            return CacheHit(
                id=str(row[0]),
                response_text=row[1],
                response_metadata=metadata,
                similarity=1.0,
                prompt_tokens=row[3] or 0,
                completion_tokens=row[4] or 0,
                total_tokens=row[5] or 0,
            )

    async def _semantic_lookup(
        self, embedding: list[float], owner_id: str, model_name: str
    ) -> Optional[CacheHit]:
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        async with self.pool.session() as session:
            result = await session.execute(
                text("""
                    SELECT id, response_text, response_metadata,
                           1 - (embedding <=> CAST(:emb AS vector)) AS similarity,
                           prompt_tokens, completion_tokens, total_tokens
                    FROM cache_entries
                    WHERE owner_id = :owner
                      AND model_name = :model
                      AND expires_at > NOW()
                    ORDER BY embedding <=> CAST(:emb AS vector)
                    LIMIT 1
                """),
                {"emb": embedding_str, "owner": owner_id, "model": model_name},
            )
            row = result.fetchone()
            if not row:
                return None

            similarity = float(row[3])
            if similarity < self.settings.min_similarity_floor:
                return None

            # Update hit count
            await session.execute(
                text("""
                    UPDATE cache_entries
                    SET hit_count = hit_count + 1, last_accessed_at = NOW()
                    WHERE id = :id
                """),
                {"id": str(row[0])},
            )
            await session.commit()

            metadata = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
            return CacheHit(
                id=str(row[0]),
                response_text=row[1],
                response_metadata=metadata,
                similarity=similarity,
                prompt_tokens=row[4] or 0,
                completion_tokens=row[5] or 0,
                total_tokens=row[6] or 0,
            )

    async def safe_store(
        self,
        context_string: str,
        embedding: list[float],
        response: ChatCompletionResponse,
        owner_id: str,
        model_name: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Fire-and-forget store. Never raises — logs errors internally."""
        try:
            await self._store(context_string, embedding, response, owner_id, model_name, ttl_seconds)
        except Exception as e:
            logger.error(f"Async cache store failed: {e}")

    async def _store(
        self,
        context_string: str,
        embedding: list[float],
        response: ChatCompletionResponse,
        owner_id: str,
        model_name: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        ttl = ttl_seconds or self.settings.default_ttl_seconds
        prompt_hash = hashlib.sha256(context_string.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        response_text = response.choices[0].message.content
        response_metadata = json.dumps({
            "finish_reason": response.choices[0].finish_reason,
            "model": response.model,
        })

        async with self.pool.session() as session:
            await session.execute(
                text("""
                    INSERT INTO cache_entries
                        (prompt_hash, prompt_text, embedding, response_text,
                         response_metadata, owner_id, model_name, expires_at,
                         prompt_tokens, completion_tokens, total_tokens)
                    VALUES
                        (:hash, :text, CAST(:emb AS vector), :resp, CAST(:meta AS jsonb),
                         :owner, :model, :expires, :ptok, :ctok, :ttok)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "hash": prompt_hash,
                    "text": context_string,
                    "emb": embedding_str,
                    "resp": response_text,
                    "meta": response_metadata,
                    "owner": owner_id,
                    "model": model_name,
                    "expires": expires_at,
                    "ptok": response.usage.prompt_tokens,
                    "ctok": response.usage.completion_tokens,
                    "ttok": response.usage.total_tokens,
                },
            )
            await session.commit()

        await self._log_event("store", owner_id, model_name, None, 0, 0, 0)
        logger.debug(f"Stored cache entry hash={prompt_hash[:8]}... owner={owner_id}")

    async def _log_event(
        self,
        event_type: str,
        owner_id: str,
        model_name: str,
        similarity: Optional[float],
        lookup_ms: float,
        embed_ms: float,
        tokens_saved: int,
    ) -> None:
        if not self.settings.enable_event_logging:
            return
        try:
            async with self.pool.session() as session:
                await session.execute(
                    text("""
                        INSERT INTO cache_events
                            (event_type, owner_id, model_name, similarity_score,
                             lookup_latency_ms, embedding_latency_ms, tokens_saved)
                        VALUES (:type, :owner, :model, :sim, :lookup, :embed, :tokens)
                    """),
                    {
                        "type": event_type,
                        "owner": owner_id,
                        "model": model_name,
                        "sim": similarity,
                        "lookup": lookup_ms,
                        "embed": embed_ms,
                        "tokens": tokens_saved,
                    },
                )
                await session.commit()
        except Exception as e:
            logger.debug(f"Event logging failed (non-critical): {e}")

    async def get_stats(self, owner_id: Optional[str] = None, hours: int = 24) -> dict:
        owner_filter = "AND owner_id = :owner" if owner_id else ""
        params: dict = {"hours": hours}
        if owner_id:
            params["owner"] = owner_id

        async with self.pool.session() as session:
            result = await session.execute(
                text(f"""
                    SELECT
                        event_type,
                        COUNT(*) as count,
                        AVG(similarity_score) FILTER (WHERE similarity_score IS NOT NULL) as avg_sim,
                        AVG(lookup_latency_ms) as avg_lookup_ms,
                        SUM(tokens_saved) as total_tokens_saved
                    FROM cache_events
                    WHERE created_at > NOW() - INTERVAL '{hours} hours'
                    {owner_filter}
                    GROUP BY event_type
                """),
                params,
            )
            rows = result.fetchall()

        stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "hit_rate": 0.0,
            "avg_similarity": 0.0,
            "avg_lookup_ms": 0.0,
            "tokens_saved": 0,
            "window_hours": hours,
        }
        for row in rows:
            if row[0] == "hit":
                stats["hits"] = row[1]
                stats["avg_similarity"] = round(float(row[2] or 0), 3)
                stats["avg_lookup_ms"] = round(float(row[3] or 0), 1)
                stats["tokens_saved"] = int(row[4] or 0)
            elif row[0] == "miss":
                stats["misses"] = row[1]
            elif row[0] == "store":
                stats["stores"] = row[1]

        total = stats["hits"] + stats["misses"]
        if total > 0:
            stats["hit_rate"] = round(stats["hits"] / total * 100, 1)

        return stats
