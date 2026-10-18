"""Tests for the DuckDB store, using an in-memory database."""

import pandas as pd
import pytest

from pl_dayahead import store

TS = pd.Timestamp("2026-01-14 23:00", tz="UTC")  # 00:00 Warsaw on 15 January
CHUNK = (pd.Timestamp("2025-12-31 23:00", tz="UTC"), pd.Timestamp("2026-01-31 23:00", tz="UTC"))


def rows(value: float, retrieved: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series": ["price_da"],
            "ts_utc": [TS],
            "resolution_min": [60],
            "value": [value],
            "retrieved_at": [pd.Timestamp(retrieved, tz="UTC")],
        }
    )


@pytest.fixture
def con():
    connection = store.connect(":memory:")
    yield connection
    connection.close()


def test_timestamps_round_trip_as_the_same_instant(con):
    store.append_raw(con, rows(100.0, "2026-02-10"))
    stored = con.execute("SELECT ts_utc FROM raw_entsoe").df()["ts_utc"].iloc[0]
    assert stored == TS


def test_revisions_are_kept_and_latest_wins(con):
    store.append_raw(con, rows(100.0, "2026-01-15"))
    store.append_raw(con, rows(101.5, "2026-02-10"))
    (n_raw,) = con.execute("SELECT count(*) FROM raw_entsoe").fetchone()
    (latest,) = con.execute("SELECT value FROM entsoe_latest").fetchone()
    assert n_raw == 2
    assert latest == 101.5


def test_missing_columns_rejected(con):
    with pytest.raises(ValueError):
        store.append_raw(con, rows(100.0, "2026-02-10").drop(columns="retrieved_at"))


def test_chunk_final_only_after_settling_period(con):
    settle = pd.Timedelta(days=7)
    log = dict(source="entsoe", query="price_da", chunk_start=CHUNK[0], chunk_end=CHUNK[1], status="ok")

    store.log_chunk(con, **log, retrieved_at=pd.Timestamp("2026-02-03", tz="UTC"))
    assert not store.chunk_is_final(con, "entsoe", "price_da", *CHUNK, settle)

    store.log_chunk(con, **log, retrieved_at=pd.Timestamp("2026-02-10", tz="UTC"))
    assert store.chunk_is_final(con, "entsoe", "price_da", *CHUNK, settle)


def test_failed_download_never_counts_as_final(con):
    store.log_chunk(
        con,
        source="entsoe",
        query="price_da",
        chunk_start=CHUNK[0],
        chunk_end=CHUNK[1],
        retrieved_at=pd.Timestamp("2026-06-01", tz="UTC"),
        status="error",
        message="HTTPError: 503",
    )
    assert not store.chunk_is_final(con, "entsoe", "price_da", *CHUNK, pd.Timedelta(days=7))