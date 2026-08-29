"""
Query Agent — Natural Language → SQL → Grounded Response
=========================================================
ARCHITECTURE NOTE:
  This agent accepts natural language questions about shipment data
  and returns answers grounded exclusively in SQLite records.

  Pipeline:
    1. User question → Groq LLM → SQL query
    2. SQL query → SQLite execution → raw rows
    3. Raw rows + question → Groq LLM → grounded natural language answer

  ANTI-HALLUCINATION DESIGN:
  - Step 2 must produce actual data before Step 3 is called
  - Step 3 LLM receives ONLY the actual SQL results, not guesses
  - If no rows returned, the answer explicitly says "no data found"
  - LLM is explicitly told: "Only answer from the data provided. Never invent."

  DATABASE SCHEMA (injected into the SQL generation prompt):
  - shipments: id, file_name, customer_id, status, extraction_confidence, created_at
  - validation_results: shipment_id, field_name, status, expected, found, confidence
  - agent_decisions: shipment_id, decision, confidence, reason, mismatch_count, uncertain_count
  - audit_logs: shipment_id, agent, event, message, created_at
"""

# pyrefly: ignore [missing-import]
from groq import Groq
from utils.llm_metrics import instrumented_groq
import json
import re
from config import Config
from logger import logger
from utils.db_utils import execute_raw_sql

# Instrumented proxy: records tokens/latency/cost per call. Forwards to Groq
# unchanged - no prompt, model or behaviour is altered. See utils/llm_metrics.py.
# This agent makes two distinct calls, labelled individually at their call sites.
groq_client = instrumented_groq("query")

# Schema description injected into the SQL generation prompt
SCHEMA_DESCRIPTION = """
SQLite Database Schema for Trade Shipment Platform:

TABLE: shipments
  id TEXT PRIMARY KEY           -- unique document ID
  file_name TEXT                -- original filename
  customer_id TEXT              -- customer identifier (nike, adidas, zara, apple, generic)
  status TEXT                   -- pipeline status: pending|extracted|validated|auto_approve|human_review|amendment_required|failed
  extraction_confidence REAL    -- average confidence 0.0-1.0
  created_at TEXT               -- ISO timestamp (UTC)
  updated_at TEXT               -- ISO timestamp (UTC)
  consignee_name TEXT           -- JSON blob: {value, confidence, source_evidence}
  hs_code TEXT                  -- JSON blob: {value, confidence, source_evidence}
  port_of_loading TEXT          -- JSON blob: {value, confidence, source_evidence}
  port_of_discharge TEXT        -- JSON blob: {value, confidence, source_evidence}
  incoterms TEXT                -- JSON blob: {value, confidence, source_evidence}
  invoice_number TEXT           -- JSON blob: {value, confidence, source_evidence}
  gross_weight TEXT             -- JSON blob: {value, confidence, source_evidence}
  description_of_goods TEXT     -- JSON blob: {value, confidence, source_evidence}

TABLE: validation_results
  shipment_id TEXT              -- references shipments.id
  field_name TEXT               -- consignee_name|hs_code|port_of_loading|...
  status TEXT                   -- match|mismatch|uncertain
  expected TEXT                 -- what was required
  found TEXT                    -- what was extracted
  confidence REAL               -- field confidence 0.0-1.0
  created_at TEXT               -- ISO timestamp

TABLE: agent_decisions
  shipment_id TEXT              -- references shipments.id
  decision TEXT                 -- auto_approve|human_review|amendment_required
  confidence REAL               -- decision confidence
  reason TEXT                   -- human-readable explanation
  mismatch_count INTEGER
  uncertain_count INTEGER
  match_count INTEGER
  created_at TEXT

TABLE: audit_logs
  shipment_id TEXT
  agent TEXT                    -- extractor|validator|router|query|system
  event TEXT                    -- start|complete|error|retry
  message TEXT
  duration_seconds REAL
  error_detail TEXT
  created_at TEXT
"""


def _build_sql_prompt(question: str) -> str:
    return f"""You are an expert SQL analyst for a trade document processing system.

{SCHEMA_DESCRIPTION}

USER QUESTION: "{question}"

Generate a single SQLite SELECT query to answer this question.

RULES:
1. Only output valid SQLite SQL — no markdown, no explanation, just the SQL
2. Use LOWER() for case-insensitive text comparisons
3. For "this week", use: datetime('now', '-7 days')
4. For "today", use: date('now')
5. For "pending reviews" or "flagged", query: agent_decisions WHERE decision='human_review'
6. For "amendments", query: agent_decisions WHERE decision='amendment_required'
7. For customer queries, match against: LOWER(shipments.customer_id)
8. Limit results to 100 rows maximum
9. JSON fields (consignee_name, hs_code, etc) are stored as JSON blobs — use json_extract() if needed:
   Example: json_extract(consignee_name, '$.value') AS consignee

Output only the SQL query, nothing else."""


def _build_answer_prompt(question: str, sql: str, rows: list) -> str:
    rows_text = json.dumps(rows[:20], indent=2, default=str)  # Limit context size
    return f"""You are a trade compliance data analyst. Answer the user's question using ONLY the provided database results.

USER QUESTION: "{question}"

SQL EXECUTED: {sql}

QUERY RESULTS ({len(rows)} rows):
{rows_text}

STRICT RULES:
1. Answer ONLY from the data above — NEVER invent numbers or facts
2. If results are empty, say "No records found matching your query."
3. Be concise — 1-3 sentences maximum
4. If the count is what they asked, state the exact number
5. Do not mention SQL or database internals in your answer
6. Format numbers clearly (e.g., "7 shipments", "3 pending reviews")

Answer:"""


def _extract_sql(text: str) -> str:
    """Extracts clean SQL from LLM output, strips markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'```sql\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    # Take only the first statement (safety)
    statements = text.split(';')
    sql = statements[0].strip() + ';'
    return sql


def run_query(question: str) -> dict:
    """
    Query Agent — Primary Entry Point.

    Returns:
        {
            question: str,
            sql_generated: str,
            row_count: int,
            answer: str,
            raw_results: list,
            success: bool
        }
    """
    logger.info(f"[Query] Received question: {question}")

    result = {
        "question": question,
        "sql_generated": "",
        "row_count": 0,
        "answer": "",
        "raw_results": [],
        "success": False
    }

    # Step 1: Generate SQL from natural language
    try:
        sql_completion = groq_client.chat.completions.create(
            _agent="query_sql",
            messages=[{"role": "user", "content": _build_sql_prompt(question)}],
            model=Config.GROQ_MODEL,
            timeout=Config.LLM_TIMEOUT_SECONDS,
            temperature=0.0,  # Fully deterministic SQL generation
            max_tokens=500
        )
        raw_sql = sql_completion.choices[0].message.content or ""
        sql = _extract_sql(raw_sql)
        result["sql_generated"] = sql
        logger.info(f"[Query] Generated SQL: {sql}")

    except Exception as e:
        logger.error(f"[Query] SQL generation failed: {e}")
        result["answer"] = "I was unable to generate a database query for your question. Please try rephrasing."
        return result

    # Step 2: Execute SQL against SQLite
    try:
        rows = execute_raw_sql(sql)
        result["raw_results"] = rows
        result["row_count"] = len(rows)
        logger.info(f"[Query] SQL returned {len(rows)} rows.")

    except ValueError as ve:
        # Only SELECT queries allowed
        logger.warning(f"[Query] SQL security violation: {ve}")
        result["answer"] = "That query type is not permitted. Only data retrieval questions are supported."
        return result
    except Exception as e:
        logger.error(f"[Query] SQL execution failed: {e}")
        result["answer"] = f"Database query failed: {str(e)}"
        return result

    # Step 3: Generate grounded natural language answer
    try:
        answer_completion = groq_client.chat.completions.create(
            _agent="query_answer",
            messages=[{"role": "user", "content": _build_answer_prompt(question, sql, rows)}],
            model=Config.GROQ_MODEL,
            timeout=Config.LLM_TIMEOUT_SECONDS,
            temperature=0.1,
            max_tokens=300
        )
        result["answer"] = (answer_completion.choices[0].message.content or "").strip()
        result["success"] = True
        logger.info(f"[Query] Answer generated successfully.")

    except Exception as e:
        logger.warning(f"[Query] Answer generation failed, using raw result: {e}")
        # Fallback: format raw results as plain text
        if not rows:
            result["answer"] = "No records found matching your query."
        else:
            result["answer"] = f"Found {len(rows)} record(s). First result: {json.dumps(rows[0], default=str)}"
        result["success"] = True

    return result
