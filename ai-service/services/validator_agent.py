"""
Validator Agent — Deterministic Field-by-Field Validation
==========================================================
ARCHITECTURE NOTE:
  This agent receives the Extractor Agent's output and validates it against
  customer-specific rules loaded from customer_rules.json.

  KEY DESIGN DECISIONS:
  1. ZERO LLM CALLS — entirely deterministic Python logic.
     Validation must be reproducible and auditable, not probabilistic.
  2. NO SILENT APPROVALS — every field gets an explicit status.
  3. Low confidence fields are automatically marked "uncertain" regardless
     of whether the value matches the rule.
  4. Three outcomes per field: match | mismatch | uncertain
     - match: value passes all applicable rules AND confidence >= threshold
     - mismatch: value found but violates a rule (shows expected vs found)
     - uncertain: value is null OR confidence below threshold
  5. Rules are loaded at module import time and cached.
"""

import json
import os
from difflib import SequenceMatcher
from config import Config
from logger import logger


def _fuzzy_similarity(a: str, b: str) -> float:
    """Returns a 0.0–1.0 similarity score between two strings (case-insensitive)."""
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()


# ── Customer Rules Cache (loaded once at startup) ──────────────────────────────
_rules_cache: dict | None = None

def _load_rules() -> dict:
    """Loads customer_rules.json. Always reloads from disk so edits are picked up instantly."""
    global _rules_cache

    rules_path = Config.CUSTOMER_RULES_PATH
    if not os.path.exists(rules_path):
        logger.error(f"[Validator] customer_rules.json not found at: {rules_path}")
        raise FileNotFoundError(f"customer_rules.json not found at: {rules_path}")

    with open(rules_path, "r") as f:
        data = json.load(f)
    _rules_cache = data
    return _rules_cache


def get_customer_rules(customer_id: str) -> dict:
    """
    Returns the rule set for a given customer_id.
    Falls back to 'generic' if customer_id is not found.
    """
    rules_data = _load_rules()
    customers = rules_data.get("customers", {})
    customer_id_lower = customer_id.lower() if customer_id else "generic"

    if customer_id_lower in customers:
        return customers[customer_id_lower]

    # Try partial match (e.g., "Nike Inc." → "nike")
    for key in customers:
        if key in customer_id_lower or customer_id_lower in key:
            logger.info(f"[Validator] Partial match: '{customer_id}' → '{key}'")
            return customers[key]

    logger.warning(f"[Validator] No rules for '{customer_id}', using generic fallback.")
    return customers.get("generic", {})


# ── Field Validation Logic ─────────────────────────────────────────────────────

def _validate_single_field(
    field_name: str,
    field_data: dict,
    rule_value,
    allowed_values: list | None,
    confidence_threshold: float
) -> dict:
    """
    Validates a single extracted field against its rule.
    Returns: {status, expected, found, confidence, reason}
    """
    found_value = field_data.get("value") if field_data else None
    confidence = float(field_data.get("confidence", 0.0)) if field_data else 0.0

    # ── UNCERTAIN: field not extracted or confidence too low ──
    if found_value is None:
        return {
            "status": "uncertain",
            "expected": _rule_display(rule_value, allowed_values),
            "found": None,
            "confidence": 0.0,
            "reason": f"Field '{field_name}' could not be extracted from the document."
        }

    if confidence < confidence_threshold:
        return {
            "status": "uncertain",
            "expected": _rule_display(rule_value, allowed_values),
            "found": found_value,
            "confidence": round(confidence, 3),
            "reason": f"Confidence {confidence:.2f} is below threshold {confidence_threshold:.2f}. Cannot confirm value."
        }

    # ── NO RULE: field is present but no rule to validate against ──
    if rule_value is None and (allowed_values is None or len(allowed_values) == 0):
        return {
            "status": "match",
            "expected": "any",
            "found": found_value,
            "confidence": round(confidence, 3),
            "reason": f"No specific rule defined for '{field_name}'. Field present with acceptable confidence."
        }

    # ── NORMALIZE for comparison ──
    found_normalized = found_value.strip().upper() if isinstance(found_value, str) else str(found_value).upper()
    rule_normalized = rule_value.strip().upper() if isinstance(rule_value, str) else str(rule_value).upper() if rule_value else None
    allowed_normalized = [v.strip().upper() for v in allowed_values] if allowed_values else None

    # ── CHECK: allowed_values list (broader match + OCR fuzzy fallback) ──
    if allowed_normalized:
        # First: exact substring match (original logic)
        match_found = any(
            allowed in found_normalized or found_normalized in allowed
            for allowed in allowed_normalized
        )
        if match_found:
            return {
                "status": "match",
                # pyrefly: ignore [no-matching-overload]
                "expected": " | ".join(allowed_values),
                "found": found_value,
                "confidence": round(confidence, 3),
                "reason": f"'{found_value}' matches allowed values."
            }

        # Second: fuzzy similarity match — catches OCR misreads (e.g. "Aphle" → "Apple")
        # A similarity >= 0.70 means the value is CLOSE but not exact → treat as uncertain
        # (needs human review), NOT a hard mismatch (which forces amendment).
        best_similarity = max(
            _fuzzy_similarity(found_normalized, allowed)
            for allowed in allowed_normalized
        )
        if best_similarity >= 0.70:
            return {
                "status": "uncertain",
                # pyrefly: ignore [no-matching-overload]
                "expected": " | ".join(allowed_values),
                "found": found_value,
                "confidence": round(confidence, 3),
                "reason": (
                    f"'{found_value}' is similar to an allowed value (similarity: {best_similarity:.0%}) "
                    f"but does not match exactly — possible OCR misread. Human review recommended."
                )
            }

        # Third: no match, not fuzzy close → hard mismatch
        return {
            "status": "mismatch",
            # pyrefly: ignore [no-matching-overload]
            "expected": " | ".join(allowed_values),
            "found": found_value,
            "confidence": round(confidence, 3),
            "reason": f"Expected one of {allowed_values}, but found '{found_value}'."
        }

    # ── CHECK: exact/contains rule_value ──
    if rule_normalized:
        if rule_normalized in found_normalized or found_normalized in rule_normalized:
            return {
                "status": "match",
                "expected": rule_value,
                "found": found_value,
                "confidence": round(confidence, 3),
                "reason": f"'{found_value}' satisfies rule '{rule_value}'."
            }
        else:
            return {
                "status": "mismatch",
                "expected": rule_value,
                "found": found_value,
                "confidence": round(confidence, 3),
                "reason": f"Expected '{rule_value}', but found '{found_value}'."
            }

    return {
        "status": "match",
        "expected": "any",
        "found": found_value,
        "confidence": round(confidence, 3),
        "reason": "Field present and within acceptable confidence."
    }


def _rule_display(rule_value, allowed_values) -> str:
    """Formats the expected value for display in validation output."""
    if allowed_values:
        return " | ".join(allowed_values)
    if rule_value:
        return str(rule_value)
    return "any"


def validate_shipment(extracted_fields: dict, customer_id: str) -> dict:
    """
    Validator Agent — Primary Entry Point.

    Args:
        extracted_fields: Output from Extractor Agent (8 fields with value/confidence/evidence)
        customer_id: Customer identifier to look up rules

    Returns:
        {
            "customer_id": str,
            "customer_name": str,
            "confidence_threshold": float,
            "field_results": {
                field_name: {status, expected, found, confidence, reason}
            },
            "summary": {
                "total_fields": int,
                "match_count": int,
                "mismatch_count": int,
                "uncertain_count": int,
                "overall_status": "pass | fail | review"
            }
        }
    """
    logger.info(f"[Validator] Starting validation for customer: {customer_id}")
    rules = get_customer_rules(customer_id)
    threshold = float(rules.get("confidence_threshold", Config.DEFAULT_CONFIDENCE_THRESHOLD))

    field_results = {}

    # ── consignee_name ──
    field_results["consignee_name"] = _validate_single_field(
        "consignee_name",
        extracted_fields.get("consignee_name", {}),
        rule_value=rules.get("required_consignee_contains"),
        allowed_values=rules.get("allowed_consignees"),
        confidence_threshold=threshold
    )

    # ── hs_code: validate by prefix if rule exists ──
    hs_rule_prefix = rules.get("required_hs_code_prefix")
    hs_field = extracted_fields.get("hs_code", {})
    if hs_rule_prefix and hs_field.get("value"):
        hs_value = str(hs_field.get("value", "")).replace(".", "").replace(" ", "")
        rule_clean = hs_rule_prefix.replace(".", "")
        hs_matches = hs_value.startswith(rule_clean)
        if float(hs_field.get("confidence", 0.0)) < threshold:
            field_results["hs_code"] = {
                "status": "uncertain",
                "expected": f"Prefix: {hs_rule_prefix}",
                "found": hs_field.get("value"),
                "confidence": round(float(hs_field.get("confidence", 0.0)), 3),
                "reason": f"Confidence below threshold {threshold}."
            }
        elif hs_matches:
            field_results["hs_code"] = {
                "status": "match",
                "expected": f"Prefix: {hs_rule_prefix}",
                "found": hs_field.get("value"),
                "confidence": round(float(hs_field.get("confidence", 0.0)), 3),
                "reason": f"HS code starts with required prefix '{hs_rule_prefix}'."
            }
        else:
            field_results["hs_code"] = {
                "status": "mismatch",
                "expected": f"Prefix: {hs_rule_prefix}",
                "found": hs_field.get("value"),
                "confidence": round(float(hs_field.get("confidence", 0.0)), 3),
                "reason": f"HS code '{hs_field.get('value')}' does not start with required prefix '{hs_rule_prefix}'."
            }
    else:
        field_results["hs_code"] = _validate_single_field(
            "hs_code", hs_field,
            rule_value=hs_rule_prefix, allowed_values=None,
            confidence_threshold=threshold
        )

    # ── port_of_loading ──
    field_results["port_of_loading"] = _validate_single_field(
        "port_of_loading",
        extracted_fields.get("port_of_loading", {}),
        rule_value=rules.get("required_port_of_loading"),
        allowed_values=rules.get("allowed_ports_of_loading"),
        confidence_threshold=threshold
    )

    # ── port_of_discharge ──
    field_results["port_of_discharge"] = _validate_single_field(
        "port_of_discharge",
        extracted_fields.get("port_of_discharge", {}),
        rule_value=rules.get("required_port_of_discharge"),
        allowed_values=rules.get("allowed_ports_of_discharge"),
        confidence_threshold=threshold
    )

    # ── incoterms ──
    field_results["incoterms"] = _validate_single_field(
        "incoterms",
        extracted_fields.get("incoterms", {}),
        rule_value=None,
        allowed_values=rules.get("required_incoterms"),
        confidence_threshold=threshold
    )

    # ── description_of_goods: presence-only check (no specific rule) ──
    field_results["description_of_goods"] = _validate_single_field(
        "description_of_goods",
        extracted_fields.get("description_of_goods", {}),
        rule_value=None, allowed_values=None,
        confidence_threshold=threshold
    )

    # ── gross_weight: presence-only check ──
    field_results["gross_weight"] = _validate_single_field(
        "gross_weight",
        extracted_fields.get("gross_weight", {}),
        rule_value=None, allowed_values=None,
        confidence_threshold=threshold
    )

    # ── invoice_number: presence-only check ──
    field_results["invoice_number"] = _validate_single_field(
        "invoice_number",
        extracted_fields.get("invoice_number", {}),
        rule_value=None, allowed_values=None,
        confidence_threshold=threshold
    )

    # ── Summary Counts ──
    match_count = sum(1 for r in field_results.values() if r["status"] == "match")
    mismatch_count = sum(1 for r in field_results.values() if r["status"] == "mismatch")
    uncertain_count = sum(1 for r in field_results.values() if r["status"] == "uncertain")
    total = len(field_results)

    if mismatch_count > 0:
        overall_status = "fail"
    elif uncertain_count > 0:
        overall_status = "review"
    else:
        overall_status = "pass"

    logger.info(
        f"[Validator] Done. Match: {match_count}, Mismatch: {mismatch_count}, "
        f"Uncertain: {uncertain_count}. Overall: {overall_status}"
    )

    return {
        "customer_id": customer_id,
        "customer_name": rules.get("customer_name", customer_id),
        "confidence_threshold": threshold,
        "field_results": field_results,
        "summary": {
            "total_fields": total,
            "match_count": match_count,
            "mismatch_count": mismatch_count,
            "uncertain_count": uncertain_count,
            "overall_status": overall_status
        }
    }
