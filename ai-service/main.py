import os

# DEBUG: Instant unbuffered output for cloud log streaming
print(">>> [TradeAI] Multi-Agent Trade Document Service Starting <<<", flush=True)

# CRITICAL: Prevent PaddleOCR/MKL deadlocks before any imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import uuid
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from logger import logger
from config import validate_config, Config

# ── Config validation on startup ─────────────────────────────────────────────
config_error = None
try:
    validate_config()
    logger.info("Configuration validated successfully.")
except EnvironmentError as ee:
    config_error = str(ee)
    logger.error(f"Config Warning: {config_error}. Service will start but may fail.")

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TradeAI — Multi-Agent Trade Document Platform",
    version="2.0.0",
    description="Extractor → Validator → Router agent pipeline for trade document processing"
)

# CORS — allows Node.js server and React frontend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    docId: str
    fileUrl: str
    fileName: Optional[str] = "document"
    customerId: Optional[str] = "generic"  # Customer for validation rules

class QueryRequest(BaseModel):
    question: str


# ── Database Initialization ────────────────────────────────────────────────────
# ARCHITECTURE: DB init happens at startup, not on first request.
# This ensures tables exist before any pipeline runs.
from utils.db_utils import init_db
init_db()
logger.info("[TradeAI] SQLite database initialized.")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint for cloud platform health checks."""
    return {
        "status": "active",
        "service": "TradeAI Multi-Agent Platform",
        "version": "2.0.0",
        "agents": ["extractor", "validator", "router", "query"]
    }

@app.get("/health")
async def health_check():
    """Detailed health endpoint."""
    return {
        "status": "ok" if not config_error else "degraded",
        "service": "tradeai",
        "error": config_error,
        "pipeline": "extractor → validator → router",
        "storage": "sqlite + s3"
    }


# ── Core Pipeline Task ────────────────────────────────────────────────────────
def trade_pipeline_task(doc_id: str, file_url: str, file_name: str, customer_id: str):
    """
    Orchestrates the full 3-agent pipeline:
      1. Download file from S3
      2. Smart text extraction (PaddleOCR + PyMuPDF — reused unchanged)
      3. Extractor Agent → 8 shipment fields with confidence
      4. Validator Agent → field-by-field validation against customer rules
      5. Router Agent → decision + reasoning
      6. Persist all steps to SQLite
      7. Notify Node.js server of status update

    OBSERVABILITY: Every step is logged to audit_logs.
    FAILURE SAFETY: Any exception marks shipment as failed — no silent failures.
    """
    # Late imports prevent startup hangs during health checks
    from utils.s3_utils import download_s3_file
    from utils.db_utils import (
        save_shipment, save_validation, save_decision,
        log_audit, mark_shipment_failed
    )
    from services.extraction_service import smart_extraction_pipeline  # REUSED UNCHANGED
    from services.ai_service import extract_shipment_fields             # Extractor Agent
    from services.validator_agent import validate_shipment              # Validator Agent
    from services.router_agent import route_decision                    # Router Agent
    from utils.llm_metrics import cost_budget, summarise, reset as reset_metrics
    from config import Config

    pipeline_start = time.time()
    temp_file = None
    reset_metrics()
    # Circuit breaker: bounds runaway LLM spend on a single document. Baseline is
    # ~$0.0008/document, so the cap is ~60x headroom — it trips on a runaway, not
    # on normal variation. Breaching it fails the document loudly.
    budget = cost_budget(Config.MAX_COST_PER_DOCUMENT_USD or None, label=doc_id)

    logger.info(f"[Pipeline] START doc_id={doc_id} customer={customer_id}")
    log_audit(doc_id, "system", "start",
              message=f"Pipeline started for {file_name}, customer: {customer_id}")

    try:
      with budget:
        # ── Step 1: Download from S3 (reused) ────────────────────────────────
        from urllib.parse import urlparse
        ext = os.path.splitext(urlparse(file_url).path)[1].lower() or ".pdf"
        temp_file = f"temp_{uuid.uuid4().hex}_{doc_id}{ext}"

        log_audit(doc_id, "system", "start", message="Downloading from S3")
        download_s3_file(file_url, temp_file)
        log_audit(doc_id, "system", "complete", message="S3 download complete")

        # ── Step 2: Smart Text Extraction (PaddleOCR + PyMuPDF — REUSED) ────
        ext_start = time.time()
        log_audit(doc_id, "extractor", "start", message="Smart extraction pipeline started")
        raw_text = smart_extraction_pipeline(temp_file)
        ext_duration = round(time.time() - ext_start, 2)
        log_audit(doc_id, "extractor", "complete",
                  message=f"Text extracted: {len(raw_text)} chars",
                  duration=ext_duration)

        # ── Step 3: Extractor Agent ───────────────────────────────────────────
        agent_start = time.time()
        log_audit(doc_id, "extractor", "start", message="LLM field extraction started")
        extraction_result = extract_shipment_fields(raw_text)
        extracted_fields = extraction_result["extracted_fields"]
        agent_duration = round(time.time() - agent_start, 2)

        log_audit(doc_id, "extractor", "complete",
                  message=f"Extracted {len(extracted_fields)} fields. Retries: {extraction_result['retry_count']}",
                  duration=agent_duration,
                  retry_count=extraction_result["retry_count"])

        # Persist extraction result
        save_shipment(
            doc_id=doc_id,
            file_name=file_name,
            file_url=file_url,
            extracted=extracted_fields,
            customer_id=customer_id,
            extraction_duration=ext_duration + agent_duration,
            raw_text_length=len(raw_text)
        )

        # ── Step 4: Validator Agent ────────────────────────────────────────────
        val_start = time.time()
        log_audit(doc_id, "validator", "start", message="Validation against customer rules started")
        validation_output = validate_shipment(extracted_fields, customer_id)
        val_duration = round(time.time() - val_start, 2)

        log_audit(doc_id, "validator", "complete",
                  message=(
                      f"Match: {validation_output['summary']['match_count']}, "
                      f"Mismatch: {validation_output['summary']['mismatch_count']}, "
                      f"Uncertain: {validation_output['summary']['uncertain_count']}"
                  ),
                  duration=val_duration)

        save_validation(doc_id, validation_output["field_results"])

        # ── Step 5: Router Agent ───────────────────────────────────────────────
        router_start = time.time()
        log_audit(doc_id, "router", "start", message="Router decision engine started")
        decision_result = route_decision(
            validation_output,
            customer_name=validation_output.get("customer_name")
        )
        router_duration = round(time.time() - router_start, 2)

        log_audit(doc_id, "router", "complete",
                  message=f"Decision: {decision_result['decision']} (confidence: {decision_result['confidence']})",
                  duration=router_duration)

        save_decision(doc_id, decision_result)

        # ── Step 6: Final Pipeline Log ─────────────────────────────────────────
        total_duration = round(time.time() - pipeline_start, 2)
        log_audit(doc_id, "system", "complete",
                  message=f"Pipeline complete in {total_duration}s. Decision: {decision_result['decision']}",
                  duration=total_duration)

        # Cost logged alongside latency, per Phase 2 requirement.
        spend = summarise()["total"]
        logger.info(
            f"[Pipeline] SUCCESS doc_id={doc_id} in {total_duration}s "
            f"→ {decision_result['decision']} "
            f"| ${spend['cost_usd']:.6f} over {spend['calls']} LLM calls "
            f"({spend['prompt_tokens']} in / {spend['completion_tokens']} out)"
        )

    except Exception as e:
        total_duration = round(time.time() - pipeline_start, 2)
        spend = summarise()["total"]
        logger.error(
            f"[Pipeline] FAILURE doc_id={doc_id}: {e} "
            f"| ${spend['cost_usd']:.6f} spent before failing"
        )
        mark_shipment_failed(doc_id, str(e))
        log_audit(doc_id, "system", "error",
                  message="Pipeline failed",
                  duration=total_duration,
                  error_detail=str(e))

    finally:
        # Always clean up temp files — no disk leaks
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/process")
async def process_document(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Triggers the 3-agent pipeline asynchronously.
    Returns immediately — pipeline runs in background.
    Node.js server polls /shipments/:id for status updates.
    """
    logger.info(f"[API] /process received: docId={request.docId}, customer={request.customerId}")
    background_tasks.add_task(
        trade_pipeline_task,
        request.docId,
        request.fileUrl,
        request.fileName or "document",
        request.customerId or "generic"
    )
    return {
        "status": "pipeline_started",
        "docId": request.docId,
        "customer": request.customerId,
        "message": "Extractor → Validator → Router pipeline running asynchronously"
    }


@app.get("/shipments")
async def list_shipments_endpoint(
    customer_id: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """Lists all processed shipments with their pipeline status and decisions."""
    from utils.db_utils import list_shipments
    shipments = list_shipments(limit=limit, customer_id=customer_id)
    return {"shipments": shipments, "count": len(shipments)}


@app.get("/shipments/{shipment_id}")
async def get_shipment_endpoint(shipment_id: str):
    """
    Returns the complete pipeline result for one shipment:
    extracted fields + validation results + agent decision + audit trail.
    """
    from utils.db_utils import get_shipment_full
    shipment = get_shipment_full(shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")
    return shipment


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Natural language query over shipment data.
    Executes: NL → SQL → SQLite → grounded answer.
    Example: "How many shipments were flagged this week?"
    """
    from services.query_agent import run_query
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question too short")
    result = run_query(request.question.strip())
    return result


@app.get("/decisions")
async def list_decisions_endpoint(limit: int = Query(50, le=200)):
    """Returns recent agent decisions across all shipments."""
    from utils.db_utils import execute_raw_sql
    rows = execute_raw_sql("""
        SELECT d.shipment_id, d.decision, d.confidence,
               d.mismatch_count, d.uncertain_count, d.match_count,
               d.created_at, s.file_name, s.customer_id
        FROM agent_decisions d
        LEFT JOIN shipments s ON s.id = d.shipment_id
        ORDER BY d.created_at DESC
        LIMIT ?
    """, (limit,))
    return {"decisions": rows, "count": len(rows)}


@app.get("/stats")
async def stats_endpoint():
    """Dashboard statistics summary."""
    from utils.db_utils import execute_raw_sql
    total = execute_raw_sql("SELECT COUNT(*) as count FROM shipments")[0]["count"]
    approved = execute_raw_sql(
        "SELECT COUNT(*) as count FROM agent_decisions WHERE decision='auto_approve'"
    )[0]["count"]
    review = execute_raw_sql(
        "SELECT COUNT(*) as count FROM agent_decisions WHERE decision='human_review'"
    )[0]["count"]
    amendment = execute_raw_sql(
        "SELECT COUNT(*) as count FROM agent_decisions WHERE decision='amendment_required'"
    )[0]["count"]
    failed = execute_raw_sql(
        "SELECT COUNT(*) as count FROM shipments WHERE status='failed'"
    )[0]["count"]

    return {
        "total_shipments": total,
        "auto_approved": approved,
        "human_review": review,
        "amendment_required": amendment,
        "failed": failed
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting TradeAI Multi-Agent Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
