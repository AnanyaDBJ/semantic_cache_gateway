"""
Automated setup for the Semantic Cache Gateway Proxy.

Usage:
    python setup_cache.py <instance-name> [--app-name NAME] [--embedding-model MODEL] [--llm-model MODEL]

Example:
    python setup_cache.py my-semantic-cache
    python setup_cache.py my-semantic-cache --app-name my-cache-app --skip-schema
"""

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path

try:
    import psycopg2
except ImportError:
    try:
        import psycopg as psycopg2
    except ImportError:
        print("Error: psycopg2 or psycopg is required. Install with: pip install 'psycopg[binary]'")
        sys.exit(1)

from databricks.sdk import WorkspaceClient

SOURCE_DIR = Path(__file__).parent
POLL_INTERVAL_LAKEBASE = 15
TIMEOUT_LAKEBASE = 600
POLL_INTERVAL_APP = 10
TIMEOUT_APP = 600
UPLOAD_EXTENSIONS = {".py", ".yaml", ".txt"}
EXCLUDED_FILES = {"setup_cache.py", "DEPLOYMENT_GUIDE.md", "CLAUDE.md"}
EXCLUDED_DIRS = {".", "__", ".git"}


def extract_config_value(content: str, field_name: str) -> str:
    pattern = rf'{field_name}:\s*str\s*=\s*"([^"]*)"'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(f"Could not find field '{field_name}' in config.py")
    return match.group(1)


def step1_create_lakebase_instance(w: WorkspaceClient, instance_name: str) -> None:
    print(f"\n[1/7] Creating Lakebase autoscale instance '{instance_name}'...")

    try:
        w.api_client.do("GET", f"/api/2.0/postgres/projects/{instance_name}")
        print(f"  Instance already exists. Skipping creation.")
        return
    except Exception:
        pass

    w.api_client.do(
        "POST",
        f"/api/2.0/postgres/projects?project_id={instance_name}",
        body={
            "name": f"projects/{instance_name}",
            "status": {
                "pg_version": 17,
                "display_name": instance_name,
            },
        },
    )
    print(f"  Instance creation initiated.")


def step2_wait_for_active(w: WorkspaceClient, instance_name: str) -> str:
    print(f"\n[2/7] Waiting for instance to become ACTIVE and retrieving host...")
    start = time.time()

    while time.time() - start < TIMEOUT_LAKEBASE:
        try:
            resp = w.api_client.do(
                "GET",
                f"/api/2.0/postgres/projects/{instance_name}/branches/production/endpoints",
            )
            endpoints = resp.get("endpoints", [])
            if endpoints:
                status = endpoints[0].get("status", {})
                hosts = status.get("hosts", {})
                host = hosts.get("host", "")
                if host:
                    print(f"  Instance ACTIVE. Host: {host}")
                    return host
        except Exception:
            pass

        elapsed = int(time.time() - start)
        sys.stdout.write(f"\r  Waiting... ({elapsed}s elapsed)")
        sys.stdout.flush()
        time.sleep(POLL_INTERVAL_LAKEBASE)

    raise TimeoutError(
        f"\n  Instance did not become ACTIVE within {TIMEOUT_LAKEBASE}s. "
        f"Re-run with --skip-lakebase-create once it's ready."
    )


def step3_setup_database_schema(w: WorkspaceClient, instance_name: str, host: str) -> None:
    print(f"\n[3/7] Setting up database schema...")

    me = w.current_user.me().user_name
    endpoint = f"projects/{instance_name}/branches/production/endpoints/primary"

    print(f"  Generating credential for {me}...")
    cred = w.postgres.generate_database_credential(endpoint=endpoint)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                database="databricks_postgres",
                user=me,
                port=5432,
                password=cred.token,
                sslmode="require",
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Connection failed ({e}), retrying in 30s...")
                time.sleep(30)
            else:
                raise RuntimeError(
                    f"Could not connect to Lakebase after {max_retries} attempts: {e}"
                )

    conn.autocommit = True
    cur = conn.cursor()

    ddl_statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE SCHEMA IF NOT EXISTS semantic_cache",
        "SET search_path TO semantic_cache, public",
        """CREATE TABLE IF NOT EXISTS cache_entries (
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
        )""",
        """CREATE TABLE IF NOT EXISTS cache_events (
            id BIGSERIAL PRIMARY KEY,
            event_type VARCHAR(20) NOT NULL,
            owner_id VARCHAR(256) DEFAULT 'global',
            model_name VARCHAR(256),
            similarity_score REAL,
            lookup_latency_ms REAL,
            embedding_latency_ms REAL,
            tokens_saved INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )""",
        "CREATE INDEX IF NOT EXISTS idx_a_prompt_hash ON cache_entries (prompt_hash, owner_id, model_name)",
        "CREATE INDEX IF NOT EXISTS idx_a_expires ON cache_entries (expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_a_owner_model ON cache_entries (owner_id, model_name, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_a_events_type_time ON cache_events (event_type, created_at DESC)",
        """CREATE INDEX IF NOT EXISTS idx_a_embedding_hnsw
            ON cache_entries
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200)""",
        "ALTER DEFAULT PRIVILEGES IN SCHEMA semantic_cache GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO PUBLIC",
        "GRANT USAGE ON SCHEMA semantic_cache TO PUBLIC",
        "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA semantic_cache TO PUBLIC",
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA semantic_cache TO PUBLIC",
    ]

    for stmt in ddl_statements:
        label = stmt.strip().split("\n")[0][:70]
        print(f"  Executing: {label}...")
        cur.execute(stmt)

    cur.close()
    conn.close()
    print(f"  Schema setup complete.")


def step4_update_config_files(instance_name: str, host: str, embedding_model: str, llm_model: str) -> None:
    print(f"\n[4/7] Updating config.py and app.yaml...")

    # --- config.py ---
    config_path = SOURCE_DIR / "config.py"
    content = config_path.read_text()

    old_instance = extract_config_value(content, "lakebase_instance_name")
    old_host = extract_config_value(content, "lakebase_host")

    content = content.replace(
        f'lakebase_instance_name: str = "{old_instance}"',
        f'lakebase_instance_name: str = "{instance_name}"',
    )
    content = content.replace(
        f'lakebase_endpoint: str = "projects/{old_instance}/branches/production/endpoints/primary"',
        f'lakebase_endpoint: str = "projects/{instance_name}/branches/production/endpoints/primary"',
    )
    content = content.replace(
        f'lakebase_host: str = "{old_host}"',
        f'lakebase_host: str = "{host}"',
    )
    if embedding_model != "databricks-gte-large-en":
        content = content.replace(
            'embedding_model: str = "databricks-gte-large-en"',
            f'embedding_model: str = "{embedding_model}"',
        )
    if llm_model != "databricks-claude-opus-4-6":
        content = content.replace(
            'default_llm_model: str = "databricks-claude-opus-4-6"',
            f'default_llm_model: str = "{llm_model}"',
        )

    config_path.write_text(content)
    print(f"  Updated config.py")

    # --- app.yaml ---
    yaml_path = SOURCE_DIR / "app.yaml"
    yaml_content = yaml_path.read_text()

    old_branch = f"projects/{old_instance}/branches/production"
    new_branch = f"projects/{instance_name}/branches/production"
    yaml_content = yaml_content.replace(old_branch, new_branch)

    if embedding_model != "databricks-gte-large-en":
        yaml_content = yaml_content.replace("databricks-gte-large-en", embedding_model)
    if llm_model != "databricks-claude-opus-4-6":
        yaml_content = yaml_content.replace("databricks-claude-opus-4-6", llm_model)

    yaml_path.write_text(yaml_content)
    print(f"  Updated app.yaml")


def step5_upload_to_workspace(w: WorkspaceClient, app_name: str) -> str:
    print(f"\n[5/7] Uploading source code to workspace...")

    me = w.current_user.me().user_name
    workspace_path = f"/Workspace/Users/{me}/apps/{app_name}/{app_name}"

    w.workspace.mkdirs(workspace_path)

    uploaded = 0
    for root, dirs, files in os.walk(SOURCE_DIR):
        dirs[:] = [d for d in dirs if not any(d.startswith(ex) for ex in EXCLUDED_DIRS)]

        for fname in files:
            if fname in EXCLUDED_FILES:
                continue
            local_path = Path(root) / fname
            if local_path.suffix not in UPLOAD_EXTENSIONS:
                continue

            relative = local_path.relative_to(SOURCE_DIR)
            remote_path = f"{workspace_path}/{relative}"

            remote_dir = str(Path(remote_path).parent)
            w.workspace.mkdirs(remote_dir)

            content_b64 = base64.b64encode(local_path.read_bytes()).decode()

            try:
                w.workspace.import_(
                    path=remote_path,
                    content=content_b64,
                    format="AUTO",
                    overwrite=True,
                )
            except Exception:
                w.api_client.do("POST", "/api/2.0/workspace/import", body={
                    "path": remote_path,
                    "content": content_b64,
                    "overwrite": True,
                    "format": "AUTO",
                })

            uploaded += 1
            print(f"  Uploaded: {relative}")

    print(f"  {uploaded} files uploaded to {workspace_path}")
    return workspace_path


def step6_create_and_deploy_app(
    w: WorkspaceClient, app_name: str, workspace_path: str,
    instance_name: str, embedding_model: str,
) -> None:
    print(f"\n[6/7] Creating and deploying app '{app_name}'...")

    from databricks.sdk.service.apps import App, AppDeployment

    try:
        app_obj = App(name=app_name, description="Semantic Cache Gateway Proxy")
        w.apps.create(app=app_obj).result()
        print(f"  App '{app_name}' created.")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "already_exists" in err:
            print(f"  App '{app_name}' already exists. Proceeding to deploy.")
        else:
            raise

    # Register resources (serving endpoint + postgres) via REST API
    print(f"  Registering resources (postgres + serving endpoint)...")
    w.api_client.do("PATCH", f"/api/2.0/apps/{app_name}", body={
        "resources": [
            {
                "name": "serving-endpoint",
                "serving_endpoint": {
                    "name": embedding_model,
                    "permission": "CAN_QUERY",
                },
            },
            {
                "name": "database",
                "postgres": {
                    "branch": f"projects/{instance_name}/branches/production",
                    "database": f"projects/{instance_name}/branches/production/databases/databricks-postgres",
                    "permission": "CAN_CONNECT_AND_CREATE",
                },
            },
        ],
    })

    deployment = AppDeployment(source_code_path=workspace_path)
    w.apps.deploy(app_name=app_name, app_deployment=deployment)
    print(f"  Deployment initiated. Waiting for completion...")

    start = time.time()
    while time.time() - start < TIMEOUT_APP:
        try:
            app_info = w.apps.get(name=app_name)
            app_state = getattr(getattr(app_info, "app_status", None), "state", None)
            active_dep = getattr(app_info, "active_deployment", None)
            deploy_state = getattr(getattr(active_dep, "status", None), "state", None)
        except Exception:
            resp = w.api_client.do("GET", f"/api/2.0/apps/{app_name}")
            app_state = resp.get("app_status", {}).get("state")
            deploy_state = resp.get("active_deployment", {}).get("status", {}).get("state")

        app_state_str = app_state.value if hasattr(app_state, "value") else str(app_state)
        deploy_state_str = deploy_state.value if hasattr(deploy_state, "value") else str(deploy_state)

        if "SUCCEEDED" in deploy_state_str and "RUNNING" in app_state_str:
            print(f"\n  App deployed and RUNNING!")
            return
        if "CRASHED" in app_state_str or "FAILED" in deploy_state_str:
            raise RuntimeError(
                f"Deployment failed: app_state={app_state_str}, deploy_state={deploy_state_str}. "
                f"Check logs with: databricks apps get {app_name}"
            )

        elapsed = int(time.time() - start)
        sys.stdout.write(f"\r  Status: app={app_state_str}, deploy={deploy_state_str} ({elapsed}s)")
        sys.stdout.flush()
        time.sleep(POLL_INTERVAL_APP)

    raise TimeoutError(f"\n  Deployment did not complete within {TIMEOUT_APP}s.")


def step7_verify(w: WorkspaceClient, app_name: str) -> None:
    print(f"\n[7/7] Deployment complete!")

    try:
        app_info = w.apps.get(name=app_name)
        url = getattr(app_info, "url", None)
    except Exception:
        resp = w.api_client.do("GET", f"/api/2.0/apps/{app_name}")
        url = resp.get("url")

    if url:
        print(f"\n  App URL:         {url}")
        print(f"  Health check:    {url}/health")
        print(f"  Chat endpoint:   {url}/v1/chat/completions")
        print(f"  Cache stats:     {url}/v1/cache/stats")
        print(f"\n  Test with:")
        print(f'    curl -H "Authorization: Bearer $(databricks auth token)" {url}/health')
    else:
        print(f"\n  App '{app_name}' deployed. Retrieve URL with:")
        print(f"    databricks apps get {app_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Semantic Cache Gateway Proxy to Databricks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("instance_name", help="Lakebase project name (e.g., 'my-semantic-cache')")
    parser.add_argument("--app-name", default=None, help="App name (defaults to instance_name)")
    parser.add_argument("--embedding-model", default="databricks-gte-large-en")
    parser.add_argument("--llm-model", default="databricks-claude-opus-4-6")
    parser.add_argument("--skip-lakebase-create", action="store_true", help="Skip instance creation")
    parser.add_argument("--skip-schema", action="store_true", help="Skip DDL setup")
    parser.add_argument("--skip-deploy", action="store_true", help="Only update config, skip upload/deploy")

    args = parser.parse_args()
    instance_name = args.instance_name
    app_name = args.app_name or instance_name

    print("=" * 60)
    print("  Semantic Cache Gateway Proxy — Automated Setup")
    print("=" * 60)
    print(f"  Instance name:   {instance_name}")
    print(f"  App name:        {app_name}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  LLM model:       {args.llm_model}")

    if len(app_name) > 30:
        print(f"\n  ERROR: App name '{app_name}' is {len(app_name)} chars (max 30).")
        print(f"  Use --app-name to specify a shorter name.")
        sys.exit(1)

    w = WorkspaceClient()
    me = w.current_user.me().user_name
    print(f"  Authenticated as: {me}")

    if not args.skip_lakebase_create:
        step1_create_lakebase_instance(w, instance_name)
    else:
        print(f"\n[1/7] Skipping Lakebase creation (--skip-lakebase-create)")

    host = step2_wait_for_active(w, instance_name)

    if not args.skip_schema:
        step3_setup_database_schema(w, instance_name, host)
    else:
        print(f"\n[3/7] Skipping schema setup (--skip-schema)")

    step4_update_config_files(instance_name, host, args.embedding_model, args.llm_model)

    if not args.skip_deploy:
        workspace_path = step5_upload_to_workspace(w, app_name)
        step6_create_and_deploy_app(w, app_name, workspace_path, instance_name, args.embedding_model)
        step7_verify(w, app_name)
    else:
        print(f"\n[5-7] Skipping upload and deploy (--skip-deploy)")

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
