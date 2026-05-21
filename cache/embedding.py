import time
import logging
from typing import Optional

from openai import AsyncOpenAI
from databricks.sdk.core import Config

from config import Settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Async client for Databricks embedding endpoints (OpenAI-compatible).
    Auto-refreshes token before expiry.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[AsyncOpenAI] = None
        self._token_expires_at: float = 0

    def _refresh_client(self) -> AsyncOpenAI:
        cfg = Config()
        headers = cfg.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        host = cfg.host.replace("https://", "").replace("http://", "")

        self._client = AsyncOpenAI(
            base_url=f"https://{host}/serving-endpoints",
            api_key=token,
            timeout=self.settings.embedding_timeout_seconds,
        )
        self._token_expires_at = time.time() + 3500
        return self._client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None or time.time() >= self._token_expires_at:
            return self._refresh_client()
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate a single embedding vector (1024 dims)."""
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self.settings.embedding_model,
                input=[text],
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding call failed, retrying with fresh token: {e}")
            client = self._refresh_client()
            response = await client.embeddings.create(
                model=self.settings.embedding_model,
                input=[text],
            )
            return response.data[0].embedding
