"""
Extractor Agent — Trade Document Field Extraction
==================================================
ARCHITECTURE NOTE:
  This module is the Extractor Agent. It takes raw text (from the
  smart_extraction_pipeline which reuses PaddleOCR + PyMuPDF) and uses
  Groq's LLaMA 3.3 70B to extract 8 structured shipment fields.

  Design decisions:
  - Each field returns {value, confidence, source_evidence} — never just a value
  - Confidence is a float 0.0–1.0 (not a string like "high/medium/low")
  - source_evidence is the exact snippet from the document that justified the value
  - The LLM is instructed to return null for uncertain fields, NEVER hallucinate
  - Rule-based detection (classify_trade_document_type) runs first as a hint
    to improve LLM accuracy on ambiguous documents
"""

# pyrefly: ignore [missing-import]
from groq import Groq
from utils.llm_metrics import instrumented_groq
import json
import re
from config import Config
from logger import logger

# Initialize Groq client (singleton — reused across requests)
# Instrumented proxy: records tokens/latency/cost per call. Forwards to Groq
# unchanged — no prompt, model or behaviour is altered. See utils/llm_metrics.py.
groq_client = instrumented_groq("extractor")

# The 8 required fields per GoComet assignment spec
REQUIRED_FIELDS = [
    "consignee_name",
    "hs_code",
    "port_of_loading",
    "port_of_discharge",
    "incoterms",
    "description_of_goods",
    "gross_weight",
    "invoice_number"
]


def classify_trade_document_type(text: str) -> str:
    """
    Fast rule-based hint to classify the trade document type.
    This hint is injected into the LLM prompt to improve field extraction accuracy.
    Does NOT replace LLM extraction — only primes it.
    """
    text_lower = text.lower()

    if "bill of lading" in text_lower or "b/l" in text_lower or "b.l." in text_lower:
        return "Bill of Lading"
    if "commercial invoice" in text_lower or "pro forma invoice" in text_lower:
        return "Commercial Invoice"
    if "packing list" in text_lower:
        return "Packing List"
    if "certificate of origin" in text_lower:
        return "Certificate of Origin"
    if "airway bill" in text_lower or "air waybill" in text_lower or "awb" in text_lower:
        return "Air Waybill"
    if "letter of credit" in text_lower or "documentary credit" in text_lower:
        return "Letter of Credit"
    if "customs declaration" in text_lower or "customs entry" in text_lower:
        return "Customs Declaration"
    if "freight invoice" in text_lower or "freight bill" in text_lower:
        return "Freight Invoice"
    if "insurance certificate" in text_lower:
        return "Insurance Certificate"
    if "inspection certificate" in text_lower:
        return "Inspection Certificate"

    return "Trade Document (Unknown Type)"


def _build_extraction_prompt(text: str, doc_hint: str) -> str:
    """
    Builds the LLM prompt for structured field extraction.
    CRITICAL RULES embedded in the prompt:
    - Never hallucinate — return null when uncertain
    - Confidence must be a decimal 0.0–1.0
    - source_evidence must be a direct quote from the document text
    """
    return f"""You are a senior Trade Document Intelligence AI specializing in international shipping and logistics.

DOCUMENT TYPE HINT: This appears to be a "{doc_hint}"

DOCUMENT TEXT (raw OCR/extracted):
\"\"\"
{text[:7000]}
\"\"\"

TASK: Extract the following 8 shipment fields from this trade document.

STRICT RULES (violations will cause data integrity failures):
1. NEVER invent or guess values — if you are not confident, set value to null
2. confidence is a decimal between 0.00 and 1.00 (not words like "high")
3. source_evidence must be a verbatim snippet from the document text (max 120 chars) that proves the value
4. If a field cannot be found, return: {{"value": null, "confidence": 0.0, "source_evidence": null}}
5. For incoterms: normalize to standard codes (FOB, CIF, CFR, EXW, DAP, DDP, etc.)
6. For hs_code: include all digits exactly as they appear (e.g., "6404.11", "8471.30.01")
7. For gross_weight: include the unit (e.g., "2450 KG", "5400 LBS")

RESPOND WITH ONLY THIS JSON (no markdown, no explanation):
{{
  "document_type": "Exact document type identified",
  "consignee_name": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "hs_code": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "port_of_loading": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "port_of_discharge": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "incoterms": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "description_of_goods": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "gross_weight": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }},
  "invoice_number": {{
    "value": "string or null",
    "confidence": 0.00,
    "source_evidence": "exact quote from document or null"
  }}
}}"""


def _validate_and_normalize_field(field_data: dict | None) -> dict:
    """
    Ensures every field output has the required structure.
    Normalizes confidence to float, clamps to [0.0, 1.0].
    This is a defensive layer — the LLM can return unexpected formats.
    """
    if not isinstance(field_data, dict):
        return {"value": None, "confidence": 0.0, "source_evidence": None}

    value = field_data.get("value")
    # Treat empty strings and "N/A" as null — not meaningful values
    if isinstance(value, str) and value.strip().lower() in ("", "n/a", "none", "unknown", "not found", "null"):
        value = None

    try:
        confidence = float(field_data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    except (TypeError, ValueError):
        confidence = 0.0

    evidence = field_data.get("source_evidence")
    if isinstance(evidence, str) and evidence.strip().lower() in ("", "n/a", "none", "null"):
        evidence = None

    # If value is None, confidence must be 0 — no false confidence on missing data
    if value is None:
        confidence = 0.0

    return {
        "value": value,
        "confidence": round(confidence, 3),
        "source_evidence": evidence
    }


def extract_shipment_fields(text: str, retry_count: int = 0) -> dict:
    """
    Extractor Agent — Primary Entry Point.

    Takes raw extracted text, runs Groq LLM to extract 8 trade shipment fields.
    Returns structured dict with each field as {value, confidence, source_evidence}.

    OBSERVABILITY: retry_count is tracked and returned for audit logging.
    FAILURE MODE: On LLM error, returns all-null fields with confidence=0 — never crashes.
    """
    # 1. Text normalization
    clean_text = re.sub(r'\s+', ' ', text).strip()

    # 2. Rule-based document type hint (fast, no API call)
    doc_hint = classify_trade_document_type(clean_text)
    logger.info(f"[Extractor] Document type hint: {doc_hint}")

    # 3. Build and send LLM prompt
    prompt = _build_extraction_prompt(clean_text, doc_hint)

    for attempt in range(Config.MAX_LLM_RETRIES + 1):
        try:
            logger.info(f"[Extractor] LLM extraction attempt {attempt + 1}...")
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=Config.GROQ_MODEL,
                response_format={"type": "json_object"},
                timeout=Config.LLM_TIMEOUT_SECONDS,
                temperature=0.1  # Low temperature for deterministic extraction
            )

            # pyrefly: ignore [bad-argument-type]
            raw_result = json.loads(completion.choices[0].message.content)
            logger.info(f"[Extractor] Raw LLM response received. Doc type: {raw_result.get('document_type')}")

            # 4. Validate and normalize each field
            extracted = {
                "document_type": raw_result.get("document_type", doc_hint)
            }
            for field in REQUIRED_FIELDS:
                extracted[field] = _validate_and_normalize_field(raw_result.get(field))

            # Log summary of what was found
            found_count = sum(1 for f in REQUIRED_FIELDS if extracted[f]["value"] is not None)
            avg_conf = sum(extracted[f]["confidence"] for f in REQUIRED_FIELDS) / len(REQUIRED_FIELDS)
            logger.info(f"[Extractor] Extraction complete: {found_count}/{len(REQUIRED_FIELDS)} fields found, avg confidence: {avg_conf:.2f}")

            return {
                "extracted_fields": extracted,
                "retry_count": attempt,
                "success": True
            }

        except json.JSONDecodeError as je:
            logger.warning(f"[Extractor] JSON parse error on attempt {attempt + 1}: {je}")
            if attempt < Config.MAX_LLM_RETRIES:
                continue
        except Exception as e:
            logger.error(f"[Extractor] LLM call failed on attempt {attempt + 1}: {e}")
            if attempt < Config.MAX_LLM_RETRIES:
                continue
            break

    # 5. Failure fallback — return all-null fields (never crash, never hallucinate)
    logger.error("[Extractor] All extraction attempts failed. Returning null fields.")
    null_field = {"value": None, "confidence": 0.0, "source_evidence": None}
    return {
        "extracted_fields": {
            "document_type": doc_hint,
            **{f: dict(null_field) for f in REQUIRED_FIELDS}
        },
        "retry_count": Config.MAX_LLM_RETRIES,
        "success": False
    }
