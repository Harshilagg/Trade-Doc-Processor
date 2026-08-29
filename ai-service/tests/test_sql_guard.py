"""
SQL guard tests — the SELECT-only security claim in the README.

The guard in db_utils.execute_raw_sql is a bare prefix check:

    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise ValueError(...)

These tests run against a throwaway SQLite file (the `temp_db` fixture), never
the real shipments.db. Behaviour below was measured, not assumed.
"""
import sqlite3

import pytest


def table_names(db):
    rows = db.execute_raw_sql(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    return {r["name"] for r in rows}


class TestWriteStatementsRejected:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO shipments (id) VALUES ('x')",
            "UPDATE shipments SET status='x'",
            "DELETE FROM shipments",
            "DROP TABLE shipments",
            "ALTER TABLE shipments ADD COLUMN evil TEXT",
            "CREATE TABLE evil (id TEXT)",
        ],
    )
    def test_write_statement_raises_value_error(self, temp_db, sql):
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            temp_db.execute_raw_sql(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "insert into shipments (id) values ('x')",
            "drop table shipments",
            "   DROP TABLE shipments",
            "\n\tDELETE FROM shipments",
        ],
    )
    def test_case_and_whitespace_variants_still_rejected(self, temp_db, sql):
        """.strip().upper() normalisation opens no bypass."""
        with pytest.raises(ValueError):
            temp_db.execute_raw_sql(sql)

    def test_rejected_write_does_not_reach_the_database(self, temp_db):
        before = table_names(temp_db)
        with pytest.raises(ValueError):
            temp_db.execute_raw_sql("DROP TABLE shipments")
        assert "shipments" in table_names(temp_db)
        assert table_names(temp_db) == before


class TestStackedStatements:
    """
    FINDING: a stacked statement PASSES the guard. "SELECT 1; DROP TABLE
    shipments" starts with SELECT, so the prefix check lets it through to
    conn.execute(). What stops it is Python's sqlite3 driver refusing
    multi-statement strings — sqlite3.ProgrammingError, not the guard's
    ValueError. The protection is incidental to the stdlib API, not something
    execute_raw_sql provides. Measured on Python 3.13.5 / SQLite 3.53.1.

    These tests assert the ProgrammingError deliberately: if a future change
    routes SQL through executescript() or a driver that permits stacked
    statements, the guard alone would not stop the DROP and these fail.
    """

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE shipments",
            "SELECT 1;DROP TABLE shipments;",
            "  select 1; drop table shipments",
            "SELECT * FROM shipments; DELETE FROM shipments",
        ],
    )
    def test_stacked_statement_is_stopped_by_driver_not_guard(self, temp_db, sql):
        with pytest.raises(sqlite3.ProgrammingError, match="one statement at a time"):
            temp_db.execute_raw_sql(sql)

    def test_stacked_drop_leaves_table_intact(self, temp_db):
        with pytest.raises(sqlite3.ProgrammingError):
            temp_db.execute_raw_sql("SELECT 1; DROP TABLE shipments")
        assert "shipments" in table_names(temp_db)

    def test_guard_itself_does_not_reject_stacked_sql(self, temp_db):
        """Pins the gap explicitly: the failure is NOT a ValueError."""
        with pytest.raises(Exception) as exc_info:
            temp_db.execute_raw_sql("SELECT 1; DROP TABLE shipments")
        assert not isinstance(exc_info.value, ValueError)


class TestLegitimateQueriesAllowed:
    def test_plain_select_returns_empty_list(self, temp_db):
        assert temp_db.execute_raw_sql("SELECT * FROM shipments") == []

    def test_parameterised_select_works_through_the_guard(self, temp_db):
        rows = temp_db.execute_raw_sql(
            "SELECT * FROM shipments WHERE customer_id = ?", ("nike",)
        )
        assert rows == []

    def test_lowercase_select_allowed(self, temp_db):
        assert temp_db.execute_raw_sql("select count(*) as c from shipments") == [
            {"c": 0}
        ]


class TestGuardOverBlocksReadOnlyQueries:
    """
    FINDING (availability, not security): the prefix check rejects read-only SQL
    that does not literally begin with SELECT. Both forms below are safe reads
    the NL->SQL step in query_agent.py can plausibly emit, and both are refused.
    Documented so a future relaxation of the guard is a deliberate change.
    """

    def test_read_only_cte_is_rejected(self, temp_db):
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            temp_db.execute_raw_sql(
                "WITH t AS (SELECT * FROM shipments) SELECT COUNT(*) AS c FROM t"
            )

    def test_comment_prefixed_select_is_rejected(self, temp_db):
        with pytest.raises(ValueError, match="Only SELECT queries are permitted"):
            temp_db.execute_raw_sql("-- count rows\nSELECT COUNT(*) AS c FROM shipments")
