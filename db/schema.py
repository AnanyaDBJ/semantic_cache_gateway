import logging
from sqlalchemy import text
from db.connection import LakebasePool

logger = logging.getLogger(__name__)


async def initialize_schema(pool: LakebasePool) -> None:
    """Verify schema exists and set search_path (DDL managed by migration notebook)."""
    async with pool.session() as session:
        # Set search_path to include semantic_cache and public (for pgvector types)
        await session.execute(text("SET search_path TO semantic_cache, public"))

        # Verify tables exist (read-only check)
        result = await session.execute(text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'semantic_cache' AND tablename = 'cache_entries'"
        ))
        if not result.fetchone():
            raise RuntimeError(
                "cache_entries table not found. "
                "Run the 'Lakebase Migration - Semantic Cache Setup' notebook first."
            )

        result = await session.execute(text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'semantic_cache' AND tablename = 'cache_events'"
        ))
        if not result.fetchone():
            raise RuntimeError(
                "cache_events table not found. "
                "Run the 'Lakebase Migration - Semantic Cache Setup' notebook first."
            )

        # Set HNSW search quality for this session
        await session.execute(text("SET hnsw.ef_search = 100"))
        await session.commit()

    logger.info("Schema verification complete \u2014 tables and indexes exist")
