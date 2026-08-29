"""
SQLite Database Utilities — Multi-Agent Trade Document Platform
===============================================================
ARCHITECTURE NOTE:
  Replaces Firestore as the shipment data store. SQLite was chosen because:
  1. Zero infrastructure — embedded, no separate server
  2. Portable — the .db file travels with the service
  3. Queryable — the Query Agent runs raw SQL against it
  4. Fast — all reads/writes are local disk I/O

  Firebase is still used for AUTH only (handled in server.js).
  Raw files remain in S3.

  Schema:
    shipments         — extracted fields per document
    validation_results — field-by-field validator output
    agent_decisions   — router decision + reasoning
    audit_logs        — every pipeline step (full observability)
"""

import sqlite3
import json
import time
from datetime import datetime
from config import Config
from logger import logger


def get_connection():
    """Returns a thread-safe SQLite connection with WAL mode for concurrency."""
    conn = sqlite3.connect(Config.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enables dict-style access
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """
    Creates all tables on startup if they don't exist.
    Called once at application startup in main.py.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # ── shipments: core extracted fields from trade documents ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS shipments (
                id TEXT PRIMARY KEY,
                file_name TEXT,
                file_url TEXT,
                customer_id TEXT DEFAULT 'generic',
                
                -- Extracted fields (stored as JSON: {value, confidence, source_evidence})
                consignee_name TEXT,
                hs_code TEXT,
                port_of_loading TEXT,
                port_of_discharge TEXT,
                incoterms TEXT,
                description_of_goods TEXT,
                gross_weight TEXT,
                invoice_number TEXT,
                
                -- Overall extraction metadata
                extraction_confidence REAL DEFAULT 0.0,
                extraction_duration_seconds REAL DEFAULT 0.0,
                raw_text_length INTEGER DEFAULT 0,
                
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # ── validation_results: field-by-field validator output ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                status TEXT NOT NULL,      -- match | mismatch | uncertain
                expected TEXT,
                found TEXT,
                confidence REAL DEFAULT 0.0,
                reason TEXT,
                created_at TEXT,
                FOREIGN KEY (shipment_id) REFERENCES shipments(id)
            )
        """)

        # ── agent_decisions: router agent output ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id TEXT NOT NULL,
                decision TEXT NOT NULL,    -- auto_approve | human_review | amendment_required
                confidence REAL DEFAULT 0.0,
                reason TEXT,              -- Human-readable explanation from LLM
                amendment_draft TEXT,     -- JSON list of amendments (if applicable)
                approval_summary TEXT,    -- Summary text (if auto-approved)
                mismatch_count INTEGER DEFAULT 0,
                uncertain_count INTEGER DEFAULT 0,
                match_count INTEGER DEFAULT 0,
                created_at TEXT,
                FOREIGN KEY (shipment_id) REFERENCES shipments(id)
            )
        """)

        # ── audit_logs: full observability, every pipeline step ──
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                shipment_id TEXT,
                agent TEXT NOT NULL,       -- extractor | validator | router | query | system
                event TEXT NOT NULL,       -- start | complete | error | retry
                message TEXT,
                duration_seconds REAL,
                retry_count INTEGER DEFAULT 0,
                error_detail TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        logger.info("[DB] SQLite schema initialized successfully.")
    except Exception as e:
        logger.error(f"[DB] Schema initialization failed: {e}")
        raise
    finally:
        conn.close()


def _now() -> str:
    # pyrefly: ignore [deprecated]
    return datetime.utcnow().isoformat()


def save_shipment(doc_id: str, file_name: str, file_url: str,
                  extracted: dict, customer_id: str,
                  extraction_duration: float, raw_text_length: int) -> bool:
    """
    Persists the Extractor Agent output to the shipments table.
    Each field is stored as a JSON blob: {value, confidence, source_evidence}.
    """
    conn = get_connection()
    try:
        # Calculate overall extraction confidence as average of field confidences
        field_names = ["consignee_name", "hs_code", "port_of_loading", "port_of_discharge",
                       "incoterms", "description_of_goods", "gross_weight", "invoice_number"]
        confidences = []
        for fn in field_names:
            field = extracted.get(fn, {})
            if field and field.get("confidence") is not None:
                confidences.append(float(field.get("confidence", 0.0)))
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        now = _now()
        conn.execute("""
            INSERT OR REPLACE INTO shipments
            (id, file_name, file_url, customer_id,
             consignee_name, hs_code, port_of_loading, port_of_discharge,
             incoterms, description_of_goods, gross_weight, invoice_number,
             extraction_confidence, extraction_duration_seconds, raw_text_length,
             status, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            doc_id, file_name, file_url, customer_id,
            json.dumps(extracted.get("consignee_name")),
            json.dumps(extracted.get("hs_code")),
            json.dumps(extracted.get("port_of_loading")),
            json.dumps(extracted.get("port_of_discharge")),
            json.dumps(extracted.get("incoterms")),
            json.dumps(extracted.get("description_of_goods")),
            json.dumps(extracted.get("gross_weight")),
            json.dumps(extracted.get("invoice_number")),
            avg_confidence, extraction_duration, raw_text_length,
            "extracted", now, now
        ))
        conn.commit()
        logger.info(f"[DB] Shipment saved: {doc_id} (avg confidence: {avg_confidence})")
        return True
    except Exception as e:
        logger.error(f"[DB] save_shipment failed for {doc_id}: {e}")
        return False
    finally:
        conn.close()


def save_validation(shipment_id: str, validation_results: dict) -> bool:
    """
    Persists Validator Agent field-by-field results.
    validation_results: { field_name: {status, expected, found, confidence, reason} }
    """
    conn = get_connection()
    try:
        now = _now()
        rows = []
        for field_name, result in validation_results.items():
            rows.append((
                shipment_id,
                field_name,
                result.get("status", "uncertain"),
                str(result.get("expected", "")) if result.get("expected") is not None else None,
                str(result.get("found", "")) if result.get("found") is not None else None,
                float(result.get("confidence", 0.0)),
                result.get("reason", ""),
                now
            ))

        conn.executemany("""
            INSERT INTO validation_results
            (shipment_id, field_name, status, expected, found, confidence, reason, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, rows)

        # Update shipment status
        conn.execute(
            "UPDATE shipments SET status='validated', updated_at=? WHERE id=?",
            (now, shipment_id)
        )
        conn.commit()
        logger.info(f"[DB] Validation results saved for: {shipment_id}")
        return True
    except Exception as e:
        logger.error(f"[DB] save_validation failed for {shipment_id}: {e}")
        return False
    finally:
        conn.close()


def save_decision(shipment_id: str, decision_result: dict) -> bool:
    """
    Persists Router Agent decision, reasoning, and any amendment draft.
    """
    conn = get_connection()
    try:
        now = _now()
        conn.execute("""
            INSERT INTO agent_decisions
            (shipment_id, decision, confidence, reason, amendment_draft,
             approval_summary, mismatch_count, uncertain_count, match_count, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            shipment_id,
            decision_result.get("decision"),
            float(decision_result.get("confidence", 0.0)),
            decision_result.get("reason", ""),
            json.dumps(decision_result.get("amendment_draft", [])),
            decision_result.get("approval_summary", ""),
            int(decision_result.get("mismatch_count", 0)),
            int(decision_result.get("uncertain_count", 0)),
            int(decision_result.get("match_count", 0)),
            now
        ))

        # Update shipment final status
        final_status = decision_result.get("decision", "human_review")
        conn.execute(
            "UPDATE shipments SET status=?, updated_at=? WHERE id=?",
            (final_status, now, shipment_id)
        )
        conn.commit()
        logger.info(f"[DB] Decision saved for: {shipment_id} → {decision_result.get('decision')}")
        return True
    except Exception as e:
        logger.error(f"[DB] save_decision failed for {shipment_id}: {e}")
        return False
    finally:
        conn.close()


def log_audit(shipment_id: str, agent: str, event: str,
              message: str = "", duration: float = 0.0,
              retry_count: int = 0, error_detail: str = "") -> None:
    """
    Appends a single observability event to audit_logs.
    ARCHITECTURE: Every pipeline step logged here prevents silent failures.
    """
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO audit_logs
            (shipment_id, agent, event, message, duration_seconds, retry_count, error_detail, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (shipment_id, agent, event, message, duration, retry_count, error_detail, _now()))
        conn.commit()
    except Exception as e:
        # Audit logging should NEVER crash the pipeline — swallow errors here
        logger.warning(f"[DB] audit_log write failed (non-fatal): {e}")
    finally:
        conn.close()


def mark_shipment_failed(shipment_id: str, error_msg: str) -> None:
    """Marks a shipment as failed and logs the error."""
    conn = get_connection()
    try:
        now = _now()
        # If shipment record doesn't exist yet, create a minimal one
        conn.execute("""
            INSERT OR IGNORE INTO shipments (id, status, created_at, updated_at)
            VALUES (?, 'failed', ?, ?)
        """, (shipment_id, now, now))
        conn.execute(
            "UPDATE shipments SET status='failed', updated_at=? WHERE id=?",
            (now, shipment_id)
        )
        conn.commit()
        log_audit(shipment_id, "system", "error", message=error_msg, error_detail=error_msg)
        logger.warning(f"[DB] Shipment marked failed: {shipment_id}")
    except Exception as e:
        logger.error(f"[DB] mark_shipment_failed error: {e}")
    finally:
        conn.close()


def get_shipment_full(shipment_id: str) -> dict | None:
    """
    Returns the complete pipeline result for one shipment:
    extracted fields + validation results + agent decision + audit trail.
    """
    conn = get_connection()
    try:
        # Core shipment
        row = conn.execute(
            "SELECT * FROM shipments WHERE id=?", (shipment_id,)
        ).fetchone()
        if not row:
            return None
        shipment = dict(row)

        # Parse JSON field blobs
        json_fields = ["consignee_name", "hs_code", "port_of_loading", "port_of_discharge",
                       "incoterms", "description_of_goods", "gross_weight", "invoice_number"]
        for f in json_fields:
            if shipment.get(f):
                try:
                    shipment[f] = json.loads(shipment[f])
                except Exception:
                    pass

        # Validation results
        vrows = conn.execute(
            "SELECT * FROM validation_results WHERE shipment_id=? ORDER BY id",
            (shipment_id,)
        ).fetchall()
        shipment["validation"] = {r["field_name"]: dict(r) for r in vrows}

        # Agent decision
        drow = conn.execute(
            "SELECT * FROM agent_decisions WHERE shipment_id=? ORDER BY id DESC LIMIT 1",
            (shipment_id,)
        ).fetchone()
        if drow:
            decision = dict(drow)
            decision["amendment_draft"] = json.loads(decision.get("amendment_draft") or "[]")
            shipment["decision"] = decision
        else:
            shipment["decision"] = None

        # Recent audit trail (last 20 events)
        audit = conn.execute(
            "SELECT * FROM audit_logs WHERE shipment_id=? ORDER BY id DESC LIMIT 20",
            (shipment_id,)
        ).fetchall()
        shipment["audit_trail"] = [dict(a) for a in audit]

        return shipment
    except Exception as e:
        logger.error(f"[DB] get_shipment_full failed: {e}")
        return None
    finally:
        conn.close()


def list_shipments(limit: int = 50, customer_id: str | None = None) -> list:
    """Returns a summary list of shipments, newest first."""
    conn = get_connection()
    try:
        if customer_id:
            rows = conn.execute("""
                SELECT s.id, s.file_name, s.customer_id, s.status,
                       s.extraction_confidence, s.created_at,
                       d.decision, d.confidence as decision_confidence
                FROM shipments s
                LEFT JOIN agent_decisions d ON d.shipment_id = s.id
                WHERE s.customer_id=?
                ORDER BY s.created_at DESC LIMIT ?
            """, (customer_id, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT s.id, s.file_name, s.customer_id, s.status,
                       s.extraction_confidence, s.created_at,
                       d.decision, d.confidence as decision_confidence
                FROM shipments s
                LEFT JOIN agent_decisions d ON d.shipment_id = s.id
                ORDER BY s.created_at DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[DB] list_shipments failed: {e}")
        return []
    finally:
        conn.close()


def execute_raw_sql(sql: str, params: tuple = ()) -> list:
    """
    Executes a raw SELECT query against the SQLite DB.
    Used exclusively by the Query Agent (NL→SQL→results pipeline).
    SECURITY: Only SELECT statements are allowed.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are permitted in the query agent.")
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[DB] execute_raw_sql failed: {e}\nSQL: {sql}")
        raise
    finally:
        conn.close()
