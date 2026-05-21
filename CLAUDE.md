# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Databricks App that proxies OpenAI-compatible `/v1/chat/completions` requests with a two-tier semantic cache backed by Lakebase (managed PostgreSQL with pgvector). Cache hits return stored responses without calling the LLM, saving cost and latency.

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
```

Requires Databricks SDK authentication configured (`~/.databrickscfg` or environment variables). In Databricks Apps, `PGHOST`/`PGUSER`/`PGDATABASE`/`PGPORT` are auto-injected.

## Deploying

One-command setup (creates Lakebase instance, runs DDL, updates config, uploads code, deploys app):

```bash
python setup_cache.py <instance-name>
python setup_cache.py my-cache --app-name my-app --skip-schema
```

The `app.yaml` defines the runtime config, environment variables, and resource bindings (Lakebase database + embedding serving endpoint). The app verifies tables exist at startup.

## Architecture

**Request flow** (`POST /v1/chat/completions`):
1. Build normalized context string from system prompt + last N user/assistant turns
2. If caching eligible (temperature=0, not streaming, no bypass header):
   - Tier 1: SHA-256 exact hash lookup against `cache_entries.prompt_hash`
   - Tier 2: Generate embedding → pgvector HNSW cosine similarity search
3. On miss: forward to Databricks FM API, then fire-and-forget store the response+embedding
4. Return OpenAI-compatible response with `X-Cache-Status` header

**Cache skip conditions**: `stream=true`, `temperature > 0`, or `X-Cache-Bypass: true` header.

**Multi-tenancy**: `X-Cache-Owner-Id` header scopes cache lookups/stores per owner. Defaults to `"global"`.

**Token management**: Both the Lakebase connection pool and the OpenAI-compatible clients (embedding + LLM) use Databricks SDK OAuth tokens that auto-refresh before expiry (~50 min cycle). The `LakebasePool` injects fresh tokens at connect-time via SQLAlchemy's `do_connect` event.

**Background tasks**:
- Token refresh loop (every 50 min)
- TTL eviction loop (every 10 min, deletes expired entries)

## Key Configuration

Settings are in `config.py` via pydantic-settings (env vars override defaults). Production values in `app.yaml` override code defaults — notably `SIMILARITY_THRESHOLD=0.92` in prod vs `0.80` in code.

## Database Schema

Two tables in `semantic_cache` schema (DDL managed externally):
- `cache_entries` — stores prompt hash, embedding (vector 1024), response, metadata, TTL, hit count
- `cache_events` — append-only event log for hit/miss/store metrics

The `search_path` is set to `semantic_cache, public` (public needed for pgvector types).

## Conventions

- All DB operations use async SQLAlchemy with raw SQL (`text()`), not ORM models
- Pydantic models in `cache/models.py` define the OpenAI-compatible request/response schema
- The embedding is computed once and reused: on miss, the embedding from lookup is passed to store to avoid a redundant API call
