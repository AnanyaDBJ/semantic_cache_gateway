import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from config import Settings
from db.connection import LakebasePool
from db.schema import initialize_schema
from cache.semantic_cache import SemanticCacheService, build_context_string
from cache.embedding import EmbeddingClient
from cache.models import ChatCompletionRequest
from gateway.llm_client import LLMClient
from middleware.observability import ObservabilityMiddleware

settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize DB, schema, clients. Shutdown: cleanup."""
    logger.info("Starting Semantic Cache Proxy...")

    pool = LakebasePool(settings)
    await pool.initialize()
    pool.start_refresh()

    await initialize_schema(pool)

    embedding_client = EmbeddingClient(settings)
    llm_client = LLMClient(settings)

    cache_service = SemanticCacheService(pool, embedding_client, settings)
    cache_service.start_eviction_loop()

    app.state.pool = pool
    app.state.cache_service = cache_service
    app.state.llm_client = llm_client

    logger.info("Semantic Cache Proxy ready.")
    yield

    logger.info("Shutting down...")
    await pool.close()


app = FastAPI(
    title="Semantic Cache Proxy",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(ObservabilityMiddleware)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with semantic caching.
    Drop-in replacement for LLM endpoints.
    """
    cache_service: SemanticCacheService = request.app.state.cache_service
    llm_client: LLMClient = request.app.state.llm_client

    owner_id = request.headers.get("X-Cache-Owner-Id", "global")
    bypass = request.headers.get("X-Cache-Bypass", "").lower() == "true"

    skip_cache = bypass or body.stream or (body.temperature and body.temperature > 0)

    embedding = None
    context = None

    if not skip_cache:
        try:
            context = build_context_string(
                [m.model_dump() for m in body.messages],
                max_turns=settings.max_context_turns,
            )
            hit, embedding = await cache_service.lookup(context, owner_id, body.model)

            if hit:
                response = hit.to_openai_response(body.model)
                return JSONResponse(
                    content=response.model_dump(exclude_none=True),
                    headers={
                        "X-Cache-Status": "hit",
                        "X-Cache-Similarity": f"{hit.similarity:.3f}",
                    },
                )
        except Exception as e:
            logger.warning(f"Cache lookup failed, falling through to LLM: {e}")

    try:
        response = await llm_client.chat_completions(body)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM endpoint error: {str(e)}")

    if not skip_cache and response.choices and context:
        if embedding is None:
            try:
                embedding = await cache_service.embedding_client.embed(context)
            except Exception as e:
                logger.warning(f"Embedding for store failed, skipping store: {e}")
                embedding = None

        if embedding is not None:
            asyncio.create_task(
                cache_service.safe_store(
                    context_string=context,
                    embedding=embedding,
                    response=response,
                    owner_id=owner_id,
                    model_name=body.model,
                )
            )

    return JSONResponse(
        content=response.model_dump(exclude_none=True),
        headers={"X-Cache-Status": "miss"},
    )


@app.get("/health")
async def health(request: Request):
    """Health check for load balancers and monitoring."""
    try:
        async with request.app.state.pool.session() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        return {"status": "healthy", "cache_db": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "cache_db": "disconnected", "error": str(e)},
        )


@app.get("/v1/cache/stats")
async def cache_stats(request: Request):
    """Cache performance metrics."""
    owner_id = request.query_params.get("owner_id")
    hours = int(request.query_params.get("hours", 24))
    return await request.app.state.cache_service.get_stats(owner_id=owner_id, hours=hours)


@app.delete("/v1/cache/clear")
async def clear_cache(request: Request):
    """Clear all cache entries for an owner."""
    owner_id = request.query_params.get("owner_id", "global")
    async with request.app.state.pool.session() as session:
        from sqlalchemy import text
        result = await session.execute(
            text("DELETE FROM cache_entries WHERE owner_id = :owner"),
            {"owner": owner_id},
        )
        count = result.rowcount
        await session.commit()
    return {"deleted": count, "owner_id": owner_id}
