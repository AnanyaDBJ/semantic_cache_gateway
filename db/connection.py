import os
import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)

from config import Settings

logger = logging.getLogger(__name__)


class LakebasePool:
    """
    Production async connection pool for Lakebase PostgreSQL.

    In Databricks Apps: uses injected PGHOST/PGUSER/PGDATABASE/PGPORT plus
    SDK-generated database credential tokens.
    Outside Apps (notebook/local): uses SDK-based OAuth token with refresh.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker] = None
        self._current_token: Optional[str] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._uses_token_auth: bool = False

    @property
    def _is_apps_environment(self) -> bool:
        return bool(os.environ.get("PGHOST"))

    async def initialize(self) -> None:
        if self._is_apps_environment:
            await self._init_from_env()
        else:
            await self._init_from_sdk()

        self._session_maker = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info(f"Lakebase pool initialized (apps_env={self._is_apps_environment})")

    async def _init_from_env(self) -> None:
        host = os.environ["PGHOST"]
        user = os.environ["PGUSER"]
        database = os.environ.get("PGDATABASE", "databricks_postgres")
        port = int(os.environ.get("PGPORT", "5432"))

        logger.info(f"Connecting to Lakebase: host={host}, user={user}, db={database}, port={port}")

        self._current_token = await asyncio.to_thread(self._generate_token)
        self._engine = self._create_engine(
            host=host,
            user=user,
            database=database,
            port=port,
        )
        self._uses_token_auth = True

    def _create_engine(self, host: str, user: str, database: str, port: int) -> AsyncEngine:
        url = URL.create(
            "postgresql+psycopg",
            username=user,
            host=host,
            port=port,
            database=database,
        )

        engine = create_async_engine(
            url,
            pool_size=self.settings.db_pool_size,
            max_overflow=self.settings.db_max_overflow,
            pool_recycle=3600,
            pool_pre_ping=True,
            connect_args={
                "sslmode": "require",
                "options": "-c search_path=semantic_cache,public",
            },
        )

        @event.listens_for(engine.sync_engine, "do_connect")
        def inject_token(dialect, conn_rec, cargs, cparams):
            cparams["password"] = self._current_token
            cparams["sslmode"] = "require"

        return engine

    async def _init_from_sdk(self) -> None:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        username = w.current_user.me().user_name
        self._current_token = await asyncio.to_thread(self._generate_token)

        self._engine = self._create_engine(
            host=self.settings.lakebase_host,
            user=username,
            database=self.settings.pgdatabase,
            port=5432,
        )
        self._uses_token_auth = True

    def _generate_token(self) -> str:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        cred = w.postgres.generate_database_credential(
            endpoint=self.settings.lakebase_endpoint,
        )
        return cred.token

    def start_refresh(self) -> None:
        if self._uses_token_auth and self._refresh_task is None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())
            logger.info("Token refresh loop started (50-min interval)")

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.token_refresh_interval)
            try:
                self._current_token = await asyncio.to_thread(self._generate_token)
                logger.info("Lakebase token refreshed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token refresh failed (will retry next cycle): {e}")

    @asynccontextmanager
    async def session(self):
        if self._session_maker is None:
            raise RuntimeError("LakebasePool is not initialized")

        async with self._session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
        if self._engine:
            await self._engine.dispose()
        logger.info("Lakebase pool closed")
