from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Cache behavior
    similarity_threshold: float = 0.80
    min_similarity_floor: float = 0.70
    max_context_turns: int = 5
    default_ttl_seconds: int = 86400
    max_cache_entries: int = 500000

    # Database (auto-injected in Databricks Apps)
    pghost: Optional[str] = None
    pgdatabase: str = "databricks_postgres"
    pguser: Optional[str] = None
    pgpassword: Optional[str] = None
    pgport: int = 5432
    lakebase_instance_name: str = "my-semantic-cache"
    lakebase_endpoint: str = "projects/my-semantic-cache/branches/production/endpoints/primary"
    lakebase_host: str = "ep-xxxxx.databricks.com"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    token_refresh_interval: int = 3000  # 50 min in seconds

    # Embedding
    embedding_model: str = "databricks-gte-large-en"
    embedding_dimensions: int = 1024
    embedding_timeout_seconds: float = 10.0

    # LLM
    default_llm_model: str = "databricks-claude-opus-4-6"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3

    # Observability
    log_level: str = "INFO"
    enable_event_logging: bool = True
    eviction_interval_seconds: int = 600

    class Config:
        env_prefix = ""
        case_sensitive = False
