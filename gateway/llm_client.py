import asyncio
import logging
import time
from typing import Optional

from openai import AsyncOpenAI
from databricks.sdk.core import Config

from config import Settings
from cache.models import ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async client for Databricks Foundation Model API.
    Supports retry with exponential backoff for rate limits.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[AsyncOpenAI] = None
        self._token_expires_at: float = 0

    def _get_client(self) -> AsyncOpenAI:
        now = time.time()
        if self._client is None or now >= self._token_expires_at:
            cfg = Config()
            headers = cfg.authenticate()
            token = headers.get("Authorization", "").replace("Bearer ", "")
            host = cfg.host.replace("https://", "").replace("http://", "")

            self._client = AsyncOpenAI(
                base_url=f"https://{host}/serving-endpoints",
                api_key=token,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,
            )
            self._token_expires_at = now + 3500
        return self._client

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Forward request to FM API with retry on rate limits."""
        client = self._get_client()
        last_error = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                kwargs = {
                    "model": request.model,
                    "messages": [m.model_dump(exclude_none=True) for m in request.messages],
                    "temperature": request.temperature or 0,
                }
                if request.max_tokens is not None:
                    kwargs["max_tokens"] = request.max_tokens
                if request.top_p is not None:
                    kwargs["top_p"] = request.top_p
                if request.stop:
                    kwargs["stop"] = request.stop

                response = await client.chat.completions.create(**kwargs)

                return ChatCompletionResponse(
                    id=response.id,
                    model=response.model,
                    created=response.created,
                    choices=[
                        Choice(
                            index=c.index,
                            message=Message(role="assistant", content=c.message.content or ""),
                            finish_reason=c.finish_reason or "stop",
                        )
                        for c in response.choices
                    ],
                    usage=Usage(
                        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                        completion_tokens=response.usage.completion_tokens if response.usage else 0,
                        total_tokens=response.usage.total_tokens if response.usage else 0,
                    ),
                    x_cache_status="miss",
                )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if attempt < self.settings.llm_max_retries and (
                    "429" in error_str or "rate" in error_str or
                    "503" in error_str or "timeout" in error_str
                ):
                    delay = 2 ** attempt
                    logger.warning(f"LLM call failed (attempt {attempt+1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    if "401" in error_str or "auth" in error_str:
                        self._client = None
                        client = self._get_client()
                else:
                    raise

        raise last_error
