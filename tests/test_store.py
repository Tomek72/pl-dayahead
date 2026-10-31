"""Tests for the DuckDB store, using an in-memory database."""

import pandas as pd
import pytest

from pl_dayahead import store

JAN = (pd.Timestamp("2025-12-31 23:00", tz="UTC"), pd.Timestamp("2026-01-31 23:00", tz="UTC"))
TS_JAN = pd.Timestamp("2026-01-14 23:00", tz="UTC")  # 00:00 Warsaw on 15 January
TS_FEB = pd.Timestamp("2026-02-14 23:00", tz="UTC")  # 00:00 Warsaw on 15 February


def rows(value: float, timestamp: pd.Timestamp = TS_JAN, series: str = "price_da") -> pd.DataFrame:
    return pd.DataFrame({
        "series": [series],
        "ts_utc": [timestamp],
        "resolution_min": [60],
        "value": [value],
        "retrieved_at": [pd.Timestamp("2026-10-01", tz="UTC")],
    })


@pytest.fixture
def con():
    connection = store.connect(":memory:")
    yield connection
    connection.close()


def stored(con) -> pd.DataFrame:
    return con.execute("SELECT series, ts_utc, value FROM entsoe_latest ORDER BY series, ts_utc").df()


def test_timestamps_round_trip_as_the_same_instant(con):
    store.replace_month(con, rows(100.0), *JAN)
    assert stored(con)["ts_utc"].iloc[0] == TS_JAN


def test_a_new_download_replaces_the_month(con):
    store.replace_month(con, rows(100.0), *JAN)
    store.replace_month(con, rows(101.5), *JAN)
    table = stored(con)
    assert len(table) == 1
    assert table["value"].iloc[0] == 101.5


def test_other_series_are_left_alone(con):
    store.replace_month(con, pd.concat([rows(100.0), rows(20000.0, series="load_da_fcst")]), *JAN)
    store.replace_month(con, rows(101.5), *JAN)  # a new price download for January
    table = stored(con).set_index("series")["value"]
    assert table["price_da"] == 101.5
    assert table["load_da_fcst"] == 20000.0


def test_other_months_are_left_alone(con):
    store.replace_month(con, rows(90.0, timestamp=TS_FEB), JAN[1], pd.Timestamp("2026-02-28 23:00", tz="UTC"))
    store.replace_month(con, rows(100.0), *JAN)
    assert list(stored(con)["value"]) == [100.0, 90.0]


def test_missing_columns_rejected(con):
    with pytest.raises(ValueError):
        store.replace_month(con, rows(100.0).drop(columns="retrieved_at"), *JAN)
