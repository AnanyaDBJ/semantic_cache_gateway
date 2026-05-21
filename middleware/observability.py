import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("semantic_cache.requests")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Track request latency and inject cache headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

        if request.url.path == "/v1/chat/completions":
            cache_status = response.headers.get("X-Cache-Status", "unknown")
            logger.info(
                f"method=POST path=/v1/chat/completions "
                f"status={response.status_code} "
                f"duration_ms={duration_ms:.1f} "
                f"cache={cache_status}"
            )

        return response
