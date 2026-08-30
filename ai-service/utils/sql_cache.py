"""
NL->SQL translation cache for the Query Agent.

Caches the generated SQL, never the result rows. On a hit the SQL is re-executed
against the live database, so the answer always reflects current data. Caching
rows would go stale the moment a document is processed; caching the translation
never does — the same question maps to the same SQL for a given schema.

Key: SHA-256(normalised_question + schema_version). schema_version is derived
from the live schema, so any change to a table or column orphans every entry
rather than serving SQL that references a column that no longer exists.
"""
import hashlib
import json
import os
import re
import threading

from config import Config
from logger import logger

_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0, "writes": 0}
_schema_version_cache = None


def reset_stats():
    for key in _stats:
        _stats[key] = 0


def get_stats():
    return dict(_stats)


def normalise_question(question):
    """
    Lowercase, collapse whitespace, drop trailing punctuation.

    Deliberately conservative: it must never merge two questions that mean
    different things. "shipments over 500kg" and "shipments under 500kg" differ
    by a word and must stay distinct keys.
    """
    text = (question or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("?!. ")


def schema_version():
    """
    Fingerprint of the live schema, from sqlite_master's CREATE statements —
    the schema db_utils.init_db() actually produced. Any DDL change alters this
    and orphans every cached entry.
    """
    global _schema_version_cache
    if _schema_version_cache is not None:
        return _schema_version_cache

    try:
        from utils.db_utils import get_connection

        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        ddl = "\n".join(r[0] for r in rows)
    except Exception as exc:
        # Without a readable schema, use a sentinel that changes per process so
        # nothing is served from a cache we cannot validate against.
        logger.warning(f"[SQLCache] schema unreadable, caching disabled: {exc}")
        return "unknown-schema"

    _schema_version_cache = hashlib.sha256(ddl.encode("utf-8")).hexdigest()[:16]
    return _schema_version_cache


def cache_key(question):
    payload = f"{normalise_question(question)}|{schema_version()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load():
    path = Config.SQL_CACHE_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"[SQLCache] read failed, treating as empty: {exc}")
        return {}


def _save(data):
    path = Config.SQL_CACHE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def get(question):
    """Cached SQL for this question, or None. Never returns cached rows."""
    if not Config.SQL_CACHE_ENABLED:
        return None
    version = schema_version()
    if version == "unknown-schema":
        return None

    with _lock:
        entry = _load().get(cache_key(question))

    if entry and entry.get("schema_version") == version:
        _stats["hits"] += 1
        logger.info(f"[SQLCache] HIT — reusing SQL, re-executing against live data")
        return entry["sql"]

    _stats["misses"] += 1
    return None


def put(question, sql):
    """Store the generated SQL for this question."""
    if not Config.SQL_CACHE_ENABLED or not sql:
        return
    version = schema_version()
    if version == "unknown-schema":
        return
    try:
        with _lock:
            data = _load()
            data[cache_key(question)] = {
                "question": normalise_question(question),
                "sql": sql,
                "schema_version": version,
            }
            _save(data)
        _stats["writes"] += 1
    except Exception as exc:
        logger.warning(f"[SQLCache] write failed (non-fatal): {exc}")


def clear():
    global _schema_version_cache
    _schema_version_cache = None
    if os.path.exists(Config.SQL_CACHE_PATH):
        os.remove(Config.SQL_CACHE_PATH)
