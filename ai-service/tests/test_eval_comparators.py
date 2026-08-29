"""
Tests for the eval harness's comparison functions.

The point of these is that eval scoring must not silently disagree with the
validator. No network, no LLM: these are pure string functions.
"""
import os
import sys

import pytest

EVAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval"
)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from run_eval import (  # noqa: E402
    canonical_decision,
    fields_match,
    normalise_hs_code,
    normalise_text,
)


class TestHsCodeMatchesValidator:
    """
    The validator normalises HS codes inline at validator_agent.py:251 rather
    than exposing a helper, so run_eval mirrors it. If that line ever changes,
    these fail.
    """

    @pytest.mark.parametrize(
        "raw",
        ["8471.30.00", "8471 30 00", "84713000", "8471.30 00", " 8471.30.00 "],
    )
    def test_mirrors_validator_normalisation(self, raw):
        validator_logic = str(raw).replace(".", "").replace(" ", "")
        assert normalise_hs_code(raw) == validator_logic.upper()

    def test_punctuation_variants_compare_equal(self):
        assert fields_match("hs_code", "8471.30.00", "84713000")[0] is True
        assert fields_match("hs_code", "847130", "847130")[0] is True

    def test_different_code_does_not_match(self):
        assert fields_match("hs_code", "847130", "847131")[0] is False


class TestExactFields:
    def test_incoterms_case_and_space_insensitive(self):
        assert fields_match("incoterms", "FOB", " fob ")[0] is True

    def test_incoterms_different_term_is_a_miss(self):
        assert fields_match("incoterms", "FOB", "CIF")[0] is False

    def test_invoice_number_one_character_off_is_a_miss(self):
        assert fields_match("invoice_number", "INV-2026-001", "INV-2026-002")[0] is False

    def test_invoice_number_punctuation_is_significant(self):
        """Exact fields must not fall through to the lenient comparison."""
        assert fields_match("invoice_number", "INV-2026-001", "INV2026001")[0] is False


class TestNormalisedFields:
    def test_case_and_whitespace_ignored(self):
        assert fields_match("consignee_name", "Nike India Pvt Ltd", "  nike india pvt ltd  ")[0] is True

    def test_punctuation_ignored(self):
        assert fields_match("consignee_name", "Apple Inc.", "Apple Inc")[0] is True

    def test_different_company_is_a_miss(self):
        assert fields_match("consignee_name", "Apple Inc.", "Adidas Inc.")[0] is False

    def test_missing_value_does_not_match_a_populated_expectation(self):
        assert fields_match("port_of_loading", "Shanghai", None)[0] is False

    def test_blank_expectation_matches_blank_extraction(self):
        """An empty label asserts the extractor should also find nothing."""
        assert fields_match("port_of_loading", "", None)[0] is True

    def test_normalise_text_collapses_internal_whitespace(self):
        assert normalise_text("Laptop   Computers") == "laptop computers"


class TestDecisionMapping:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Auto approval", "auto_approve"),
            ("auto_approve", "auto_approve"),
            ("Human Review", "human_review"),
            ("human_review", "human_review"),
            ("Amendment Required", "amendment_required"),
            ("amendment_required", "amendment_required"),
            ("  AMENDMENT REQUIRED  ", "amendment_required"),
        ],
    )
    def test_prose_labels_map_to_router_constants(self, label, expected):
        assert canonical_decision(label) == expected

    def test_unrecognised_label_raises_rather_than_scoring_a_miss(self):
        """A typo must abort the run, not silently count as a wrong decision."""
        with pytest.raises(ValueError, match="Unrecognised expected_decision"):
            canonical_decision("approve maybe")
