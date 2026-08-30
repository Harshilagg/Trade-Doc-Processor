"""
Tests for the OCR text cache and the NL->SQL translation cache.

No network, no PaddleOCR, no LLM: these exercise the cache layers directly.
"""
import pytest

from utils import ocr_cache, sql_cache


@pytest.fixture
def ocr_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_cache.Config, "OCR_CACHE_DIR", str(tmp_path / "ocr"))
    monkeypatch.setattr(ocr_cache.Config, "OCR_CACHE_ENABLED", True)
    ocr_cache.reset_stats()
    return tmp_path


@pytest.fixture
def sql_tmp(tmp_path, monkeypatch, temp_db):
    monkeypatch.setattr(sql_cache.Config, "SQL_CACHE_PATH", str(tmp_path / "sql.json"))
    monkeypatch.setattr(sql_cache.Config, "SQL_CACHE_ENABLED", True)
    sql_cache._schema_version_cache = None
    sql_cache.reset_stats()
    yield tmp_path
    sql_cache._schema_version_cache = None


class TestOcrCache:
    def test_miss_then_hit_returns_identical_text(self, ocr_tmp):
        doc = ocr_tmp / "scan.pdf"
        doc.write_bytes(b"%PDF-1.4 fake bytes")

        assert ocr_cache.get(str(doc)) is None
        ocr_cache.put(str(doc), "EXTRACTED TEXT")
        assert ocr_cache.get(str(doc)) == "EXTRACTED TEXT"
        assert ocr_cache.get_stats()["hits"] == 1

    def test_key_is_content_not_filename(self, ocr_tmp):
        """Same bytes under a different name still hits."""
        a = ocr_tmp / "a.pdf"
        b = ocr_tmp / "b.pdf"
        a.write_bytes(b"identical bytes")
        b.write_bytes(b"identical bytes")

        ocr_cache.put(str(a), "TEXT")
        assert ocr_cache.get(str(b)) == "TEXT"

    def test_edited_document_misses(self, ocr_tmp):
        """A changed document must not serve the old text."""
        doc = ocr_tmp / "scan.pdf"
        doc.write_bytes(b"version one")
        ocr_cache.put(str(doc), "OLD TEXT")

        doc.write_bytes(b"version two")
        assert ocr_cache.get(str(doc)) is None

    def test_disabled_cache_never_serves(self, ocr_tmp, monkeypatch):
        doc = ocr_tmp / "scan.pdf"
        doc.write_bytes(b"bytes")
        ocr_cache.put(str(doc), "TEXT")

        monkeypatch.setattr(ocr_cache.Config, "OCR_CACHE_ENABLED", False)
        assert ocr_cache.get(str(doc)) is None

    def test_unreadable_file_does_not_raise(self, ocr_tmp):
        """A cache failure must degrade to real OCR, never break extraction."""
        assert ocr_cache.get(str(ocr_tmp / "missing.pdf")) is None

    def test_empty_string_is_cacheable_but_none_is_not(self, ocr_tmp):
        doc = ocr_tmp / "scan.pdf"
        doc.write_bytes(b"bytes")
        ocr_cache.put(str(doc), None)
        assert ocr_cache.get(str(doc)) is None

        ocr_cache.put(str(doc), "")
        assert ocr_cache.get(str(doc)) == ""


class TestSqlCacheKeying:
    def test_trivial_variants_share_a_key(self):
        base = sql_cache.cache_key("How many shipments are there?")
        for variant in [
            "how many shipments are there",
            "  How Many Shipments Are There?  ",
            "HOW MANY SHIPMENTS ARE THERE!",
            "How  many   shipments are there",
        ]:
            assert sql_cache.cache_key(variant) == base

    @pytest.mark.parametrize(
        "a,b",
        [
            ("shipments over 500kg", "shipments under 500kg"),
            ("shipments to Mumbai", "shipments from Mumbai"),
            ("how many shipments", "how many customers"),
            ("approved shipments", "rejected shipments"),
        ],
    )
    def test_opposite_questions_never_collide(self, a, b):
        """
        The exact tier must never merge questions that mean different things.
        'over 500kg' and 'under 500kg' differ by one word and mean opposites; a
        false hit there would return confidently wrong SQL.
        """
        assert sql_cache.cache_key(a) != sql_cache.cache_key(b)


class TestSqlCacheBehaviour:
    def test_miss_then_hit(self, sql_tmp):
        assert sql_cache.get("how many shipments") is None
        sql_cache.put("how many shipments", "SELECT COUNT(*) FROM shipments")
        assert sql_cache.get("how many shipments") == "SELECT COUNT(*) FROM shipments"

    def test_it_caches_sql_not_rows(self, sql_tmp):
        """
        The stored entry holds the translation only. Rows would go stale the
        moment a document is processed; SQL re-executed against live data
        cannot.
        """
        sql_cache.put("q", "SELECT COUNT(*) FROM shipments")
        import json

        with open(sql_cache.Config.SQL_CACHE_PATH, encoding="utf-8") as fh:
            stored = json.load(fh)
        entry = next(iter(stored.values()))
        assert set(entry) == {"question", "sql", "schema_version"}
        assert "rows" not in entry and "results" not in entry

    def test_schema_change_orphans_entries(self, sql_tmp, temp_db):
        sql_cache.put("how many shipments", "SELECT COUNT(*) FROM shipments")
        assert sql_cache.get("how many shipments") is not None

        conn = temp_db.get_connection()
        conn.execute("ALTER TABLE shipments ADD COLUMN newcol TEXT")
        conn.commit()
        conn.close()
        sql_cache._schema_version_cache = None

        assert sql_cache.get("how many shipments") is None

    def test_schema_version_is_stable_without_ddl_change(self, sql_tmp):
        first = sql_cache.schema_version()
        sql_cache._schema_version_cache = None
        assert sql_cache.schema_version() == first

    def test_disabled_cache_never_serves(self, sql_tmp, monkeypatch):
        sql_cache.put("q", "SELECT 1")
        monkeypatch.setattr(sql_cache.Config, "SQL_CACHE_ENABLED", False)
        assert sql_cache.get("q") is None

    def test_corrupt_cache_file_is_survivable(self, sql_tmp):
        with open(sql_cache.Config.SQL_CACHE_PATH, "w", encoding="utf-8") as fh:
            fh.write("{ not valid json")
        assert sql_cache.get("anything") is None

    def test_empty_sql_is_not_stored(self, sql_tmp):
        sql_cache.put("q", "")
        assert sql_cache.get("q") is None


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("How many shipments?", "how many shipments"),
            ("  spaced   out  ", "spaced out"),
            ("Trailing dots...", "trailing dots"),
            ("", ""),
        ],
    )
    def test_normalise_question(self, raw, expected):
        assert sql_cache.normalise_question(raw) == expected

    def test_normalisation_preserves_meaning_words(self):
        """Normalisation must not strip words that carry meaning."""
        assert "over" in sql_cache.normalise_question("shipments over 500kg")
        assert "under" in sql_cache.normalise_question("shipments under 500kg")
