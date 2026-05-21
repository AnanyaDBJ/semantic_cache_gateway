# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Databricks App that proxies OpenAI-compatible `/v1/chat/completions` requests with a two-tier semantic cache backed by Lakebase (managed PostgreSQL with pgvector). Cache hits return stored responses without calling the LLM, saving cost and latency.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires Databricks SDK auth + Lakebase instance)
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info

# Deploy everything from scratch (creates Lakebase, schema, app)
python setup_cache.py <instance-name>
python setup_cache.py my-cache --app-name my-app --embedding-model databricks-gte-large-en

# Deploy with partial steps
python setup_cache.py my-cache --skip-lakebase-create  # instance already exists
python setup_cache.py my-cache --skip-schema           # schema already set up
python setup_cache.py my-cache --skip-deploy           # only update config locally
```

## Architecture

**Request flow** (`POST /v1/chat/completions`):
1. Build normalized context string from system prompt + last N user/assistant turns (`build_context_string` in `cache/semantic_cache.py`)
2. If caching eligible (temperature=0, not streaming, no bypass header):
   - Tier 1: SHA-256 exact hash lookup against `cache_entries.prompt_hash`
   - Tier 2: Generate embedding → pgvector HNSW cosine similarity search (`1 - (embedding <=> vector)`)
3. On miss: forward to Databricks FM API (`gateway/llm_client.py`), then fire-and-forget store the response+embedding
4. Return OpenAI-compatible response with `X-Cache-Status` header

**Cache skip conditions**: `stream=true`, `temperature > 0`, or `X-Cache-Bypass: true` header.

**Multi-tenancy**: `X-Cache-Owner-Id` header scopes cache lookups/stores per owner. Defaults to `"global"`.

**Token management**: Both the Lakebase connection pool and the OpenAI-compatible clients (embedding + LLM) use Databricks SDK OAuth tokens that auto-refresh before expiry (~50 min cycle). The `LakebasePool` in `db/connection.py` injects fresh tokens at connect-time via SQLAlchemy's `do_connect` event.

**Background tasks** (started in `app.py` lifespan):
- Token refresh loop (every 50 min) — `db/connection.py:LakebasePool._refresh_loop`
- TTL eviction loop (every 10 min) — `cache/semantic_cache.py:SemanticCacheService._eviction_loop`

## Key Files

| File | What to know |
|------|-------------|
| `app.py` | FastAPI app, lifespan init, all HTTP endpoints |
| `config.py` | All settings via pydantic-settings (env vars override defaults) |
| `cache/semantic_cache.py` | Core cache logic: `lookup()`, `_exact_lookup()`, `_semantic_lookup()`, `_store()` |
| `cache/embedding.py` | Async OpenAI-compatible embedding client with token refresh |
| `cache/models.py` | Pydantic models: `ChatCompletionRequest`, `ChatCompletionResponse`, `CacheHit` |
| `db/connection.py` | `LakebasePool` — async SQLAlchemy pool with OAuth token injection |
| `db/schema.py` | Startup verification that tables exist (no DDL — managed by `setup_cache.py`) |
| `gateway/llm_client.py` | LLM forwarding with exponential backoff on 429/503 |
| `setup_cache.py` | Automated deployment: creates Lakebase, DDL, uploads, deploys app |
| `app.yaml` | Databricks App manifest (command, env, resource bindings) |

## Database Schema

Two tables in `semantic_cache` schema on Lakebase:

**`cache_entries`** — the cache store
- `prompt_hash` (VARCHAR 64) — SHA-256 of normalized context string
- `embedding` (vector 1024) — pgvector column with HNSW index (cosine ops, m=16, ef_construction=200)
- `response_text`, `response_metadata` (JSONB) — cached LLM response
- `owner_id`, `model_name` — multi-tenant scoping
- `expires_at` — TTL expiration, cleaned by eviction loop

**`cache_events`** — observability
- `event_type` (hit/miss/store), latency metrics, tokens saved

Search path: `semantic_cache, public` (public needed for pgvector types).

## Conventions

- All DB operations use async SQLAlchemy with raw SQL via `text()` — no ORM models
- The embedding is computed once and reused: on miss, the embedding from `lookup()` is passed to `store()` to avoid a redundant embedding API call
- `setup_cache.py` modifies `config.py` and `app.yaml` in-place using regex/string replacement (fields are unique enough for safe substitution)
- The `postgres` resource type in `app.yaml` is required for Lakebase Autoscale (not `database` — that's for provisioned instances only)
- Resource bindings must be registered via REST API PATCH after app creation (SDK `w.apps.create` doesn't pick them up from `app.yaml` automatically)

## Common Modifications

**Change similarity threshold**: `config.py:similarity_threshold` (code default) or `app.yaml` env `SIMILARITY_THRESHOLD` (prod override)

**Add a new endpoint**: Add route in `app.py`, access cache/pool via `request.app.state.*`

**Change embedding model**: Update `config.py:embedding_model` and `config.py:embedding_dimensions`, also update `app.yaml` serving endpoint resource name. The HNSW index DDL uses `vector(1024)` — if dimensions change, recreate the index.

**Change cache key strategy**: Modify `build_context_string()` in `cache/semantic_cache.py` — this determines what gets hashed and embedded.
