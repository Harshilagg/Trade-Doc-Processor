"""
Router Agent — Decision Engine with LLM Reasoning
===================================================
ARCHITECTURE NOTE:
  This agent reads the Validator Agent output and makes one of three decisions:
    1. auto_approve   — all fields matched with sufficient confidence
    2. human_review   — uncertain fields require a human to verify
    3. amendment_required — mismatches found, document must be corrected

  DECISION RULES (deterministic — applied before LLM):
    - Any mismatch         → amendment_required
    - Any uncertain field  → human_review (if no mismatches)
    - All matched          → auto_approve

  The LLM (Groq) is used ONLY to generate human-readable text outputs:
    - reason: natural language explanation of the decision
    - amendment_draft: list of specific corrections required
    - approval_summary: confirmation summary for auto-approved shipments

  This hybrid design ensures the decision itself is deterministic and auditable,
  while the explanation is rich and human-readable.
"""

# pyrefly: ignore [missing-import]
from groq import Groq
import json
from config import Config
from logger import logger

groq_client = Groq(api_key=Config.GROQ_API_KEY)

# Decision outcome constants
DECISION_APPROVE = "auto_approve"
DECISION_REVIEW = "human_review"
DECISION_AMEND = "amendment_required"


def _apply_decision_rules(validation_summary: dict) -> tuple[str, float]:
    """
    Pure deterministic decision logic.
    Returns (decision, base_confidence).
    This runs BEFORE the LLM — the LLM only explains the decision.
    """
    mismatch_count = validation_summary.get("mismatch_count", 0)
    uncertain_count = validation_summary.get("uncertain_count", 0)
    match_count = validation_summary.get("match_count", 0)
    total = validation_summary.get("total_fields", 8)

    # Rule 1: Any mismatch → amendment required (highest priority)
    if mismatch_count > 0:
        confidence = round(0.95 - (mismatch_count * 0.05), 2)
        return DECISION_AMEND, max(confidence, 0.70)

    # Rule 2: Any uncertainty → human review
    if uncertain_count > 0:
        confidence = round(0.85 - (uncertain_count * 0.05), 2)
        return DECISION_REVIEW, max(confidence, 0.60)

    # Rule 3: All matched → auto-approve
    if match_count == total and total > 0:
        confidence = 0.97
        return DECISION_APPROVE, confidence

    # Edge case: empty results → human review
    return DECISION_REVIEW, 0.50


def _build_reasoning_prompt(
    decision: str,
    customer_name: str,
    field_results: dict,
    summary: dict
) -> str:
    """Builds the LLM prompt to generate human-readable reasoning."""

    mismatches = {k: v for k, v in field_results.items() if v["status"] == "mismatch"}
    uncertain = {k: v for k, v in field_results.items() if v["status"] == "uncertain"}
    matched = {k: v for k, v in field_results.items() if v["status"] == "match"}

    decision_label = {
        DECISION_APPROVE: "AUTO-APPROVE (store and process)",
        DECISION_REVIEW: "HUMAN REVIEW REQUIRED",
        DECISION_AMEND: "AMENDMENT REQUIRED (document must be corrected)"
    }.get(decision, decision)

    prompt = f"""You are a Senior Trade Compliance Officer writing a formal shipment review report.

CUSTOMER: {customer_name}
DECISION: {decision_label}

VALIDATION RESULTS:
- Matched fields ({summary['match_count']}): {list(matched.keys())}
- Mismatched fields ({summary['mismatch_count']}): {json.dumps(mismatches, indent=2)}
- Uncertain fields ({summary['uncertain_count']}): {json.dumps(uncertain, indent=2)}

Generate a JSON response with EXACTLY these fields:

{{
  "reason": "3-4 sentence professional explanation of why this decision was made. Reference specific field names and values. Be specific and actionable.",
  "amendment_draft": [
    {{
      "field": "field_name",
      "issue": "what is wrong",
      "required_correction": "exactly what needs to change",
      "priority": "high | medium | low"
    }}
  ],
  "approval_summary": "If decision is auto_approve: 1-2 sentence confirmation. Otherwise: empty string."
}}

RULES:
- amendment_draft should have one entry per mismatched field (empty array if no mismatches)
- reason must be professional, specific, and mention the customer name
- Do NOT use generic phrases like "please review" — be specific about what failed
"""
    return prompt


def route_decision(validation_output: dict, customer_name: str | None = None) -> dict:
    """
    Router Agent — Primary Entry Point.

    Args:
        validation_output: Full output from validate_shipment()
        customer_name: Override customer name for reasoning prompt

    Returns:
        {
            decision: auto_approve | human_review | amendment_required,
            confidence: float,
            reason: str,
            amendment_draft: list,
            approval_summary: str,
            mismatch_count: int,
            uncertain_count: int,
            match_count: int
        }
    """
    summary = validation_output.get("summary", {})
    field_results = validation_output.get("field_results", {})
    c_name = customer_name or validation_output.get("customer_name", "Unknown Customer")

    # Step 1: Deterministic decision (no LLM)
    decision, base_confidence = _apply_decision_rules(summary)
    logger.info(f"[Router] Deterministic decision: {decision} (confidence: {base_confidence})")

    # Step 2: LLM for human-readable outputs only
    reason = ""
    amendment_draft = []
    approval_summary = ""

    try:
        prompt = _build_reasoning_prompt(decision, c_name, field_results, summary)
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=Config.GROQ_MODEL,
            response_format={"type": "json_object"},
            timeout=Config.LLM_TIMEOUT_SECONDS,
            temperature=0.2
        )
        # pyrefly: ignore [bad-argument-type]
        llm_out = json.loads(completion.choices[0].message.content)
        reason = llm_out.get("reason", "")
        amendment_draft = llm_out.get("amendment_draft", [])
        approval_summary = llm_out.get("approval_summary", "")
        logger.info(f"[Router] LLM reasoning generated. Amendment items: {len(amendment_draft)}")

    except Exception as e:
        # LLM failure for reasoning is non-fatal — decision is already made deterministically
        logger.warning(f"[Router] LLM reasoning failed (non-fatal): {e}")
        # Generate fallback reasoning text
        reason = _fallback_reasoning(decision, summary, field_results, c_name)
        amendment_draft = _fallback_amendments(field_results)
        approval_summary = (
            f"All {summary.get('match_count', 0)} fields validated successfully for {c_name}."
            if decision == DECISION_APPROVE else ""
        )

    return {
        "decision": decision,
        "confidence": base_confidence,
        "reason": reason,
        "amendment_draft": amendment_draft,
        "approval_summary": approval_summary,
        "mismatch_count": summary.get("mismatch_count", 0),
        "uncertain_count": summary.get("uncertain_count", 0),
        "match_count": summary.get("match_count", 0)
    }


def _fallback_reasoning(decision: str, summary: dict, field_results: dict, customer: str) -> str:
    """Generates deterministic fallback reasoning when LLM is unavailable."""
    mismatches = [(k, v) for k, v in field_results.items() if v["status"] == "mismatch"]
    uncertain = [(k, v) for k, v in field_results.items() if v["status"] == "uncertain"]

    if decision == DECISION_AMEND:
        issues = "; ".join(
            f"'{k}' expected '{v['expected']}' but found '{v['found']}'"
            for k, v in mismatches
        )
        return (
            f"Shipment for {customer} requires amendment due to {len(mismatches)} field mismatch(es). "
            f"Issues: {issues}. Document must be corrected before processing."
        )
    elif decision == DECISION_REVIEW:
        fields = ", ".join(k for k, _ in uncertain)
        return (
            f"Shipment for {customer} requires human review. "
            f"{len(uncertain)} field(s) could not be verified with sufficient confidence: {fields}. "
            f"A compliance officer must verify these fields before approval."
        )
    else:
        return (
            f"All {summary.get('match_count', 0)} required fields for {customer} have been "
            f"validated and match the expected values. Shipment approved for processing."
        )


def _fallback_amendments(field_results: dict) -> list:
    """Generates fallback amendment list when LLM is unavailable."""
    amendments = []
    for field, result in field_results.items():
        if result["status"] == "mismatch":
            amendments.append({
                "field": field,
                "issue": f"Value '{result['found']}' does not match required '{result['expected']}'",
                "required_correction": f"Update {field} to '{result['expected']}'",
                "priority": "high"
            })
    return amendments
