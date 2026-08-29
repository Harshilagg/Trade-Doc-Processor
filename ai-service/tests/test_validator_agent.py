"""
Validator Agent tests — deterministic, zero LLM calls.

Rule values are read from the real customer_rules.json, not a fixture copy, so
these tests fail if the shipped rules drift.
"""
import pytest

from services.validator_agent import (
    _fuzzy_similarity,
    get_customer_rules,
    validate_shipment,
)

NIKE_THRESHOLD = 0.75
GENERIC_THRESHOLD = 0.60


def field(value, confidence):
    return {"value": value, "confidence": confidence, "source_evidence": "test"}


def complete_nike_fields(**overrides):
    """All 8 fields valid for nike at high confidence; override to isolate one."""
    fields = {
        "consignee_name": field("Nike Imports LLC", 0.95),
        "hs_code": field("8471.30.00", 0.95),
        "port_of_loading": field("Shanghai", 0.95),
        "port_of_discharge": field("Los Angeles", 0.95),
        "incoterms": field("FOB", 0.95),
        "description_of_goods": field("Athletic footwear", 0.95),
        "gross_weight": field("1200 KG", 0.95),
        "invoice_number": field("INV-2024-001", 0.95),
    }
    fields.update(overrides)
    return fields


def status_of(fields, customer_id, field_name):
    return validate_shipment(fields, customer_id)["field_results"][field_name]["status"]


class TestRulesLoading:
    def test_real_rules_file_has_expected_customers(self):
        for customer_id in ("nike", "adidas", "zara", "apple", "maersk", "generic"):
            assert get_customer_rules(customer_id)["customer_id"] == customer_id

    def test_nike_threshold_matches_shipped_rules(self):
        assert get_customer_rules("nike")["confidence_threshold"] == NIKE_THRESHOLD

    def test_unknown_customer_falls_back_to_generic(self):
        rules = get_customer_rules("totally_unknown_corp")
        assert rules["customer_id"] == "generic"
        assert rules["confidence_threshold"] == GENERIC_THRESHOLD
        assert rules["required_incoterms"] is None

    def test_partial_customer_id_resolves_to_real_customer(self):
        # "nike" is a substring of "nike inc.", so this hits the partial-match
        # loop rather than the generic fallback.
        assert get_customer_rules("Nike Inc.")["customer_id"] == "nike"


class TestIncoterms:
    def test_allowed_incoterm_matches(self):
        assert status_of(complete_nike_fields(), "nike", "incoterms") == "match"

    def test_disallowed_incoterm_mismatches(self):
        fields = complete_nike_fields(incoterms=field("EXW", 0.95))
        assert status_of(fields, "nike", "incoterms") == "mismatch"


class TestConfidenceThresholdBoundary:
    """The check is `confidence < threshold`, so exactly-at-threshold passes."""

    @pytest.mark.parametrize(
        "confidence,expected",
        [(0.74, "uncertain"), (NIKE_THRESHOLD, "match"), (0.76, "match")],
    )
    def test_boundary(self, confidence, expected):
        fields = complete_nike_fields(incoterms=field("FOB", confidence))
        assert status_of(fields, "nike", "incoterms") == expected

    def test_low_confidence_overrides_a_correct_value(self):
        fields = complete_nike_fields(hs_code=field("8471.30.00", 0.50))
        assert status_of(fields, "nike", "hs_code") == "uncertain"


class TestPortAllowList:
    def test_allowed_port_matches(self):
        assert status_of(complete_nike_fields(), "nike", "port_of_loading") == "match"

    def test_unrelated_port_mismatches(self):
        # Rotterdam scores 0.2222 against the nearest nike port, well under the
        # 0.70 fuzzy cutoff, so it is a hard mismatch rather than uncertain.
        fields = complete_nike_fields(port_of_loading=field("Rotterdam", 0.95))
        assert status_of(fields, "nike", "port_of_loading") == "mismatch"


class TestConsigneeFuzzyMatching:
    """
    Ratios below are measured, not assumed. The cutoff is a hardcoded
    best_similarity >= 0.70, reached only after exact/substring matching fails.
    """

    def test_exact_consignee_matches_via_substring(self):
        assert status_of(complete_nike_fields(), "nike", "consignee_name") == "match"

    def test_near_miss_is_uncertain_not_mismatch(self):
        # "Aphle Imports LLC" scores 0.7879 against "Nike Imports LLC".
        fields = complete_nike_fields(consignee_name=field("Aphle Imports LLC", 0.95))
        result = validate_shipment(fields, "nike")["field_results"]["consignee_name"]
        assert result["status"] == "uncertain"
        assert "possible OCR misread" in result["reason"]

    def test_different_company_is_mismatch(self):
        # "Adidas Global" scores 0.5000 against the nearest nike consignee.
        fields = complete_nike_fields(consignee_name=field("Adidas Global", 0.95))
        assert status_of(fields, "nike", "consignee_name") == "mismatch"

    @pytest.mark.parametrize(
        "candidate,ratio,expected",
        [
            ("Aphle Imprts LL", 0.7097, "uncertain"),
            ("Aphle Impo LLC", 0.6667, "mismatch"),
        ],
    )
    def test_cutoff_is_bracketed(self, candidate, ratio, expected):
        """Two strings straddling 0.70, pinning which side each lands on."""
        allowed = get_customer_rules("nike")["allowed_consignees"]
        measured = max(_fuzzy_similarity(candidate.upper(), a.upper()) for a in allowed)
        assert measured == pytest.approx(ratio, abs=0.001)

        fields = complete_nike_fields(consignee_name=field(candidate, 0.95))
        assert status_of(fields, "nike", "consignee_name") == expected

    def test_shared_corporate_suffix_lifts_unrelated_name_over_cutoff(self):
        """
        FINDING, not a desired behaviour: "Zzzz Imports LLC" is an unrelated
        company but scores 0.7500 against "Nike Imports LLC" on the shared
        " Imports LLC" suffix alone, so it is reported as a possible OCR misread
        rather than a mismatch. Documented here so a future rule change to the
        cutoff or the comparison surfaces as a failure.
        """
        allowed = get_customer_rules("nike")["allowed_consignees"]
        measured = max(
            _fuzzy_similarity("ZZZZ IMPORTS LLC", a.upper()) for a in allowed
        )
        assert measured == pytest.approx(0.75, abs=0.001)

        fields = complete_nike_fields(consignee_name=field("Zzzz Imports LLC", 0.95))
        assert status_of(fields, "nike", "consignee_name") == "uncertain"


class TestHsCodePrefix:
    def test_prefix_match_ignores_punctuation(self):
        # nike requires prefix "847130"; "8471.30.00" normalises to "84713000".
        assert status_of(complete_nike_fields(), "nike", "hs_code") == "match"

    def test_wrong_prefix_mismatches(self):
        fields = complete_nike_fields(hs_code=field("6404.11.00", 0.95))
        assert status_of(fields, "nike", "hs_code") == "mismatch"


class TestMissingFields:
    def test_absent_field_is_uncertain(self):
        fields = complete_nike_fields()
        del fields["consignee_name"]
        result = validate_shipment(fields, "nike")["field_results"]["consignee_name"]
        assert result["status"] == "uncertain"
        assert "could not be extracted" in result["reason"]

    def test_null_value_is_uncertain(self):
        fields = complete_nike_fields(invoice_number=field(None, 0.0))
        assert status_of(fields, "nike", "invoice_number") == "uncertain"


class TestNullRuleFields:
    def test_maersk_null_port_rule_accepts_any_port(self):
        # maersk has both required_port_of_loading and allowed_ports_of_loading
        # null, so any present value above threshold takes the no-rule branch.
        fields = complete_nike_fields(port_of_loading=field("Reykjavik", 0.95))
        result = validate_shipment(fields, "maersk")["field_results"]["port_of_loading"]
        assert result["status"] == "match"
        assert result["expected"] == "any"

    def test_generic_customer_passes_any_populated_field(self):
        fields = complete_nike_fields(
            port_of_loading=field("Anywhere", 0.95),
            incoterms=field("XYZ", 0.95),
        )
        summary = validate_shipment(fields, "unknown_corp")["summary"]
        assert summary["mismatch_count"] == 0
        assert summary["overall_status"] == "pass"


class TestSummaryRollup:
    def test_all_match_is_pass(self):
        summary = validate_shipment(complete_nike_fields(), "nike")["summary"]
        assert summary["total_fields"] == 8
        assert summary["match_count"] == 8
        assert summary["overall_status"] == "pass"

    def test_any_uncertain_is_review(self):
        fields = complete_nike_fields(gross_weight=field("1200 KG", 0.10))
        summary = validate_shipment(fields, "nike")["summary"]
        assert summary["uncertain_count"] == 1
        assert summary["mismatch_count"] == 0
        assert summary["overall_status"] == "review"

    def test_mismatch_outranks_uncertain(self):
        fields = complete_nike_fields(
            incoterms=field("EXW", 0.95),          # mismatch
            gross_weight=field("1200 KG", 0.10),   # uncertain
        )
        summary = validate_shipment(fields, "nike")["summary"]
        assert summary["mismatch_count"] == 1
        assert summary["uncertain_count"] == 1
        assert summary["overall_status"] == "fail"
