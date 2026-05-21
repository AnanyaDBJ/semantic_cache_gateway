# Semantic Cache Gateway Proxy — Deployment Guide

Deploy the Semantic Cache Gateway Proxy app on Databricks from scratch.

---

## Prerequisites

- Databricks workspace with Apps enabled
- Access to create Lakebase (PostgreSQL) instances
- A serving endpoint for embeddings (e.g., `databricks-gte-large-en`)

---

## Step 1: Create an Autoscale Lakebase Instance

Create a new autoscale Lakebase PostgreSQL instance via the Databricks UI or CLI.

**Via UI:** Navigate to **SQL > Lakebase** and create a new project.

**Via CLI / SDK:**

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.api_client.do("POST", "/api/2.0/postgres/projects", body={
    "name": "projects/<your-instance-name>",
    "status": {
        "pg_version": 17,
        "display_name": "<your-instance-name>"
    }
})
```

Wait until the instance state is `ACTIVE`, then note down:
- **Instance name** (e.g., `semantic-cache-fresh`)
- **Host** (e.g., `ep-xxxxx.database.us-east-2.cloud.databricks.com`)
- **Endpoint path**: `projects/<instance-name>/branches/production/endpoints/primary`

You can retrieve these via:

```bash
databricks api get /api/2.0/postgres/projects/<instance-name>/branches/production/endpoints
```

---

## Step 2: Set Up the Database Schema

Connect as your admin user and create the required schema, tables, and indexes.

```python
import psycopg2
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me().user_name

# Generate a credential token
cred = w.postgres.generate_database_credential(
    endpoint="projects/<instance-name>/branches/production/endpoints/primary"
)

conn = psycopg2.connect(
    host="<your-lakebase-host>",
    database="databricks_postgres",
    user=me,
    port=5432,
    password=cred.token,
    sslmode="require"
)
conn.autocommit = True
cur = conn.cursor()

# Enable pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Create schema
cur.execute("CREATE SCHEMA IF NOT EXISTS semantic_cache")
cur.execute("SET search_path TO semantic_cache, public")

# Create tables
cur.execute("""
    CREATE TABLE IF NOT EXISTS cache_entries (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        prompt_hash VARCHAR(64) NOT NULL,
        prompt_text TEXT NOT NULL,
        embedding vector(1024) NOT NULL,
        response_text TEXT NOT NULL,
        response_metadata JSONB DEFAULT '{}',
        owner_id VARCHAR(256) NOT NULL DEFAULT 'global',
        model_name VARCHAR(256) NOT NULL,
        hit_count INTEGER DEFAULT 0,
        last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS cache_events (
        id BIGSERIAL PRIMARY KEY,
        event_type VARCHAR(20) NOT NULL,
        owner_id VARCHAR(256) DEFAULT 'global',
        model_name VARCHAR(256),
        similarity_score REAL,
        lookup_latency_ms REAL,
        embedding_latency_ms REAL,
        tokens_saved INTEGER DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW()
    )
""")

# Create performance indexes
cur.execute("CREATE INDEX IF NOT EXISTS idx_a_prompt_hash ON cache_entries (prompt_hash, owner_id, model_name)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_a_expires ON cache_entries (expires_at)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_a_owner_model ON cache_entries (owner_id, model_name, created_at DESC)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_a_events_type_time ON cache_events (event_type, created_at DESC)")

# Create HNSW vector index
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_a_embedding_hnsw
        ON cache_entries
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
""")

# Grant permissions to PUBLIC (so any platform-provisioned SP role gets access)
cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA semantic_cache GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO PUBLIC")
cur.execute("GRANT USAGE ON SCHEMA semantic_cache TO PUBLIC")
cur.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA semantic_cache TO PUBLIC")
cur.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA semantic_cache TO PUBLIC")

cur.close()
conn.close()
print("Schema setup complete!")
```

> **IMPORTANT:** Do NOT use `CREATE ROLE` for the app's service principal. The platform will auto-create the SP's role with proper `LAKEBASE_OAUTH_V1` authentication when the resource binding is set up. Manually creating a role causes authentication failures.

---

## Step 3: Configure the App Source Code

Update `config.py` with your Lakebase instance details:

```python
# config.py
class Settings(BaseSettings):
    # ...
    pgdatabase: str = "databricks_postgres"
    lakebase_instance_name: str = "<your-instance-name>"
    lakebase_endpoint: str = "projects/<your-instance-name>/branches/production/endpoints/primary"
    lakebase_host: str = "<your-lakebase-host>"
    # ...
```

Ensure `db/connection.py` uses the **SDK method** for token generation:

```python
def _generate_token(self) -> str:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    cred = w.postgres.generate_database_credential(
        endpoint=self.settings.lakebase_endpoint,
    )
    return cred.token
```

> **WARNING:** Do NOT use `w.api_client.do("POST", "/api/2.0/postgres/credentials", ...)`. This raw REST endpoint does not respect `CAN_CONNECT_AND_CREATE` permissions granted via app resource bindings and will cause authentication failures for service principals.

---

## Step 4: Configure `app.yaml`

Use the **`postgres` resource type** (not `database`) for autoscale Lakebase instances:

```yaml
command:
  - "uvicorn"
  - "app:app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"
  - "--log-level"
  - "info"

env:
  - name: SIMILARITY_THRESHOLD
    value: "0.92"
  - name: MAX_CONTEXT_TURNS
    value: "5"
  - name: DEFAULT_TTL_SECONDS
    value: "86400"
  - name: EMBEDDING_MODEL
    value: "databricks-gte-large-en"
  - name: DEFAULT_LLM_MODEL
    value: "databricks-claude-opus-4-6"
  - name: LOG_LEVEL
    value: "INFO"
  - name: DB_POOL_SIZE
    value: "5"
  - name: DB_MAX_OVERFLOW
    value: "10"

resources:
  - name: serving-endpoint
    serving_endpoint:
      name: databricks-gte-large-en
      permission: CAN_QUERY
  - name: database
    postgres:
      branch: projects/<your-instance-name>/branches/production
      database: projects/<your-instance-name>/branches/production/databases/databricks-postgres
      permission: CAN_CONNECT_AND_CREATE
```

> **CRITICAL:** For autoscale Lakebase, you MUST use the `postgres` resource type. The `database` resource type only works for provisioned (fixed-capacity) instances and will silently fail with "Database instance does not exist" for autoscale instances.

---

## Step 5: Upload Source Code to Workspace

Upload your app source code to a Workspace directory:

```bash
# Using Databricks CLI
databricks workspace import-dir ./semantic-cache-gateway-proxy-v2 \
  /Workspace/Users/<your-email>/apps/<app-name>/<app-name> \
  --overwrite
```

Or via the Databricks UI: navigate to **Workspace > Users > your folder** and upload.

---

## Step 6: Create and Deploy the App

**Create the app (first time only):**

```bash
databricks apps create --name <app-name> --output JSON
```

**Deploy:**

```bash
databricks apps deploy <app-name> \
  --source-code-path /Workspace/Users/<your-email>/apps/<app-name>/<app-name>
```

Wait for the deployment to complete:

```bash
databricks apps get <app-name> --output JSON
```

Look for:
- `app_status.state`: `RUNNING`
- `active_deployment.status.state`: `SUCCEEDED`

---

## Step 7: Verify

Once the app is running, verify connectivity:

```bash
# Check the app URL (will redirect to login if not authenticated)
curl -L https://<app-name>-<workspace-id>.aws.databricksapps.com/health
```

---

## Troubleshooting

### "password authentication failed for user '<sp-client-id>'"

This error means the service principal cannot authenticate to Lakebase. Common causes:

| Cause | Fix |
|-------|-----|
| Using `database` resource type for autoscale Lakebase | Switch to `postgres` resource type in app.yaml |
| Manually created role via `CREATE ROLE` in a notebook | Delete the role and redeploy — let the platform auto-provision it |
| Using raw REST API (`/api/2.0/postgres/credentials`) for token generation | Switch to SDK: `w.postgres.generate_database_credential()` |
| Corrupted role state on the Lakebase instance | Create a fresh Lakebase instance |

### "Database instance does not exist" when adding resource

You're using the `database` resource type with an autoscale instance. Use `postgres` instead:

```yaml
resources:
  - name: database
    postgres:
      branch: projects/<instance>/branches/production
      database: projects/<instance>/branches/production/databases/databricks-postgres
      permission: CAN_CONNECT_AND_CREATE
```

### App crashes immediately after "started successfully"

Check logs for the actual error. The deployment can report `SUCCEEDED` (container started) while the app crashes during initialization (e.g., DB connection failure). Look at `app_status.state` — if it says `CRASHED`, the startup code failed.

---

## Key Differences: Provisioned vs Autoscale Lakebase

| | Provisioned | Autoscale |
|---|---|---|
| **app.yaml resource type** | `database` | `postgres` |
| **Database name** | User-defined | Always `databricks_postgres` |
| **Resource binding format** | `instance_name` + `database_name` | `branch` + `database` (full paths) |
| **Role provisioning** | Automatic via `database` binding | Automatic via `postgres` binding |
| **Manual `CREATE ROLE`** | Not recommended | **NEVER** — breaks OAuth auth |

---

## Architecture Summary

```
Client Request
     │
     ▼
┌─────────────────────────┐
│  Databricks App (FastAPI)│
│  - Gateway Proxy         │
│  - Semantic Cache Layer  │
└─────────┬───────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌──────────────────┐
│Embedding│  │ Lakebase (PG +   │
│ Model   │  │ pgvector HNSW)   │
│ Endpoint│  │ - cache_entries  │
└─────────┘  │ - cache_events   │
             └──────────────────┘
```

The app authenticates to Lakebase using short-lived OAuth tokens generated via the Databricks SDK, refreshed every 50 minutes. The `postgres` resource binding in app.yaml injects `PGHOST`, `PGUSER`, `PGDATABASE`, and `PGPORT` environment variables and grants the SP `CAN_CONNECT_AND_CREATE` permission on the Lakebase project.
