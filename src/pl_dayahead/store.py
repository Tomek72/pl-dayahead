"""DuckDB storage.

Raw tables are append-only: every download is inserted with its retrieved_at time
and nothing is ever updated or deleted. Revisions therefore sit side by side, and
the entsoe_latest view picks the most recent copy of each interval.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from pl_dayahead.config import DB_PATH

RAW_COLUMNS = ["series", "ts_utc", "resolution_min", "value", "retrieved_at"]

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS raw_entsoe (
        series          VARCHAR     NOT NULL,  -- price_da, load_da_fcst, solar_da_fcst, ...
        ts_utc          TIMESTAMPTZ NOT NULL,  -- interval start
        resolution_min  INTEGER     NOT NULL,  -- 15, 30 or 60
        value           DOUBLE      NOT NULL,
        retrieved_at    TIMESTAMPTZ NOT NULL   -- when this copy was downloaded
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        source        VARCHAR     NOT NULL,
        query         VARCHAR     NOT NULL,
        chunk_start   TIMESTAMPTZ NOT NULL,
        chunk_end     TIMESTAMPTZ NOT NULL,
        retrieved_at  TIMESTAMPTZ NOT NULL,
        status        VARCHAR     NOT NULL,  -- ok, empty, conflict, error
        n_rows        INTEGER     NOT NULL,
        n_missing     INTEGER     NOT NULL,
        n_duplicate   INTEGER     NOT NULL,
        message       VARCHAR
    )
    """,
    """
    CREATE OR REPLACE VIEW entsoe_latest AS
    SELECT series, ts_utc, resolution_min, value, retrieved_at
    FROM raw_entsoe
    QUALIFY row_number() OVER (PARTITION BY series, ts_utc ORDER BY retrieved_at DESC) = 1
    """,
]


def connect(path: Path | str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the database (":memory:" for tests) and make sure the schema exists."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone = 'UTC'")  # timestamps come back in UTC, never local time
    for statement in SCHEMA:
        con.execute(statement)
    return con


def append_raw(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> int:
    """Insert long-format rows into raw_entsoe. Returns the number inserted."""
    missing = set(RAW_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"rows lack columns: {sorted(missing)}")
    if rows.empty:
        return 0
    cols = ", ".join(RAW_COLUMNS)
    con.register("incoming", rows[RAW_COLUMNS])
    try:
        con.execute(f"INSERT INTO raw_entsoe ({cols}) SELECT {cols} FROM incoming")
    finally:
        con.unregister("incoming")
    return len(rows)


def log_chunk(
    con: duckdb.DuckDBPyConnection,
    *,
    source: str,
    query: str,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    retrieved_at: pd.Timestamp,
    status: str,
    n_rows: int = 0,
    n_missing: int = 0,
    n_duplicate: int = 0,
    message: str | None = None,
) -> None:
    """Record one download attempt, successful or not."""
    con.execute(
        "INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            source,
            query,
            chunk_start.to_pydatetime(),
            chunk_end.to_pydatetime(),
            retrieved_at.to_pydatetime(),
            status,
            n_rows,
            n_missing,
            n_duplicate,
            message,
        ],
    )


def chunk_is_final(
    con: duckdb.DuckDBPyConnection,
    source: str,
    query: str,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    settle: pd.Timedelta,
) -> bool:
    """True if the chunk was downloaded successfully at least `settle` after it ended.

    Such a chunk is not expected to change, so it is skipped on later runs. Recent
    chunks keep being downloaded so late publications and revisions are captured.
    """
    (count,) = con.execute(
        """
        SELECT count(*) FROM ingest_log
        WHERE source = ? AND query = ? AND chunk_start = ? AND status = 'ok'
          AND retrieved_at >= ?
        """,
        [source, query, chunk_start.to_pydatetime(), (chunk_end + settle).to_pydatetime()],
    ).fetchone()
    return count > 0