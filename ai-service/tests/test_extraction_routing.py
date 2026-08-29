"""
Extraction routing tests — the digital-PDF vs OCR decision.

validate_text_quality is pure deterministic logic that chooses between two
completely different extraction paths. Thresholds below are measured against
the real code, not taken from the README.

PyMuPDF and PaddleOCR are stubbed out at import: no PDF is parsed, no OCR
engine is constructed, no network is touched.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub the heavy/optional deps before importing the module under test.
# extraction_service does `import fitz` and
# `from services.ocr_service import process_scanned_document` at module scope;
# services.ocr_service pulls in paddleocr and cv2, which we never want in a
# unit test. No other test module imports either of these.
sys.modules.setdefault("fitz", MagicMock())
if "services.ocr_service" not in sys.modules:
    _ocr_stub = types.ModuleType("services.ocr_service")
    _ocr_stub.process_scanned_document = lambda path: "<OCR-STUB>"
    sys.modules["services.ocr_service"] = _ocr_stub

from services import extraction_service  # noqa: E402
from services.extraction_service import (  # noqa: E402
    smart_extraction_pipeline,
    validate_text_quality,
)

MIN_CHARS = 80
MIN_ALPHA_RATIO = 0.25


def text_of(length, alpha_count):
    """Exactly `alpha_count` letters padded with digits to `length`."""
    assert alpha_count <= length
    return "a" * alpha_count + "1" * (length - alpha_count)


class TestCharacterCountBoundary:
    """The check is `len(text.strip()) < 80`, so exactly 80 passes."""

    @pytest.mark.parametrize(
        "length,expected",
        [(79, False), (MIN_CHARS, True), (81, True)],
    )
    def test_boundary(self, length, expected):
        assert validate_text_quality("a" * length) is expected

    def test_length_is_measured_after_stripping(self):
        """79 letters plus padding whitespace is still too short."""
        assert validate_text_quality("   " + "a" * 79 + "   ") is False


class TestAlphabeticRatioBoundary:
    """
    The check is `alpha_chars < len(text) * 0.25`, so exactly 25% passes.
    Note the code comment at extraction_service.py:12 says "must be > 25%",
    but the implemented behaviour is >= 25%. The README's ">= 25%" is correct.
    """

    @pytest.mark.parametrize(
        "alpha_count,expected",
        [(24, False), (25, True), (26, True)],
    )
    def test_boundary_at_length_100(self, alpha_count, expected):
        assert validate_text_quality(text_of(100, alpha_count)) is expected


class TestBothConditionsFail:
    def test_too_short_and_too_few_letters(self):
        # 40 chars (under 80) and 4 letters out of 40 (10%, under 25%).
        assert validate_text_quality(text_of(40, 4)) is False

    def test_short_circuits_on_length_first(self):
        """A short all-alpha string fails on length despite a perfect ratio."""
        assert validate_text_quality("a" * 40) is False


class TestRatioDenominatorIsUnstripped:
    """
    FINDING: the two checks disagree about which length they measure. The
    length check uses len(text.strip()) but the ratio divides by len(text),
    unstripped. So identical extracted text flips from the digital fast path
    to the OCR path purely by gaining surrounding whitespace, which is exactly
    what varies between PDF text layers.
    """

    def test_identical_content_flips_path_on_whitespace_padding(self):
        core = text_of(100, 25)  # exactly at the 25% ratio
        assert validate_text_quality(core) is True

        padded = "   " + core + "   "
        # Same 25 letters and same stripped length, but the ratio denominator
        # grew from 100 to 106, dropping it under the cutoff.
        assert len(padded.strip()) == len(core)
        assert validate_text_quality(padded) is False


class TestRealisticSamples:
    def test_digital_pdf_text_takes_fast_path(self):
        sample = (
            "COMMERCIAL INVOICE\n"
            "Invoice Number: INV-2024-001\n"
            "Consignee: Nike Imports LLC\n"
            "Port of Loading: Shanghai\n"
            "Port of Discharge: Los Angeles\n"
            "Incoterms: FOB\n"
            "HS Code: 8471.30.00\n"
            "Description: Athletic footwear\n"
            "Gross Weight: 1200 KG\n"
        )
        assert validate_text_quality(sample) is True

    def test_ocr_garbage_routes_to_ocr(self):
        """Mostly punctuation and digits, as a failed text layer looks."""
        garbage = "|||...---___12345 678|||...---___90123 456|||...---___789 0123|||...---___4567 89|||...---"
        # Long enough to clear the length check, so this isolates the ratio check.
        assert len(garbage) >= MIN_CHARS
        assert validate_text_quality(garbage) is False


class TestEmptyAndWhitespaceInput:
    """Reported as measured: both are rejected, so both fall through to OCR."""

    @pytest.mark.parametrize("text", ["", "   ", "\n\t  \n", " " * 200])
    def test_empty_and_whitespace_are_rejected(self, text):
        assert validate_text_quality(text) is False

    def test_whitespace_longer_than_the_threshold_still_rejected(self):
        """200 whitespace chars exceed 80 raw, but strip() to 0."""
        assert len(" " * 200) > MIN_CHARS
        assert validate_text_quality(" " * 200) is False


class TestPipelineRouting:
    """smart_extraction_pipeline's path selection, with both paths stubbed."""

    @pytest.fixture
    def spy(self, monkeypatch):
        calls = {"ocr": [], "digital": []}

        def fake_ocr(path):
            calls["ocr"].append(path)
            return "<OCR-RESULT>"

        monkeypatch.setattr(extraction_service, "process_scanned_document", fake_ocr)
        return calls

    @pytest.mark.parametrize(
        "filename", ["scan.png", "scan.jpg", "scan.jpeg", "SCAN.PNG"]
    )
    def test_image_files_go_straight_to_ocr(self, spy, monkeypatch, filename):
        def must_not_run(path):
            raise AssertionError("digital extraction attempted on an image")

        monkeypatch.setattr(extraction_service, "extract_digital_text", must_not_run)

        assert smart_extraction_pipeline(filename) == "<OCR-RESULT>"
        assert spy["ocr"] == [filename]

    def test_good_digital_text_returns_without_ocr(self, spy, monkeypatch):
        good = "Commercial Invoice for Nike Imports LLC shipping footwear " * 3
        monkeypatch.setattr(
            extraction_service, "extract_digital_text", lambda path: good
        )

        assert smart_extraction_pipeline("doc.pdf") == good
        assert spy["ocr"] == []

    def test_poor_digital_text_falls_back_to_ocr(self, spy, monkeypatch):
        monkeypatch.setattr(
            extraction_service, "extract_digital_text", lambda path: "too short"
        )

        assert smart_extraction_pipeline("doc.pdf") == "<OCR-RESULT>"
        assert spy["ocr"] == ["doc.pdf"]

    def test_failed_digital_extraction_falls_back_to_ocr(self, spy, monkeypatch):
        """extract_digital_text returns "" on error, which fails the check."""
        monkeypatch.setattr(extraction_service, "extract_digital_text", lambda path: "")

        assert smart_extraction_pipeline("doc.pdf") == "<OCR-RESULT>"
        assert spy["ocr"] == ["doc.pdf"]
