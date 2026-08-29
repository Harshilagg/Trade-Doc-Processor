"""
Shared fixtures for the deterministic test suite.

This suite makes zero network calls and needs no real credentials: no Groq API,
no AWS/S3, no PaddleOCR. It covers only pure-Python logic (validator rules,
router decision precedence) and a throwaway SQLite file (SQL guard).

Run it from the ai-service/ directory:  cd ai-service && pytest tests/ -v
Services use flat imports (`from config import Config`), which only resolve with
ai-service/ on sys.path. Running from the repo root fails at collection.
"""
import os
import sys
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent

# router_agent.py:30 and query_agent.py:34 construct Groq(api_key=...) at module
# scope, and the constructor raises on a None key rather than deferring to first
# use. Without this, importing router_agent to test its pure decision function
# crashes at collection. The value is never used for a request.
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key-not-used")

if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

import pytest


@pytest.fixture(autouse=True)
def _rules_path_anchored_to_repo(monkeypatch):
    """Resolve customer_rules.json by absolute path so tests pass from any cwd."""
    import config

    monkeypatch.setattr(
        config.Config,
        "CUSTOMER_RULES_PATH",
        str(AI_SERVICE_ROOT / "customer_rules.json"),
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A throwaway SQLite DB with the real schema, so tests never touch shipments.db."""
    import utils.db_utils as db_utils

    monkeypatch.setattr(
        db_utils.Config, "SQLITE_DB_PATH", str(tmp_path / "guard_test.db")
    )
    db_utils.init_db()
    return db_utils
