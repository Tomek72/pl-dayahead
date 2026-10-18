"""Tests for the ENTSO-E ingest helpers. No network or API token needed."""

from datetime import date

import numpy as np
import pandas as pd

from pl_dayahead.ingest.entsoe import month_chunks, series_name, to_long
from pl_dayahead.timeutils import delivery_index

RETRIEVED = pd.Timestamp("2026-10-06 12:00", tz="UTC")
JAN_2026 = month_chunks(date(2026, 1, 1), date(2026, 2, 1))[0]


def warsaw_series(day: date, freq: str = "h", value: float = 100.0) -> pd.Series:
    """Fake entsoe-py output: values indexed in Warsaw local time."""
    idx = delivery_index(day, freq).tz_convert("Europe/Warsaw")
    return pd.Series(value, index=idx)


def test_month_chunks_follow_warsaw_midnight():
    (lo, hi), = month_chunks(date(2023, 1, 1), date(2023, 2, 1))
    assert lo == pd.Timestamp("2022-12-31 23:00", tz="UTC")
    assert hi == pd.Timestamp("2023-01-31 23:00", tz="UTC")


def test_month_chunk_across_autumn_dst():
    (lo, hi), = month_chunks(date(2026, 10, 1), date(2026, 11, 1))
    assert lo == pd.Timestamp("2026-09-30 22:00", tz="UTC")  # CEST start
    assert hi == pd.Timestamp("2026-10-31 23:00", tz="UTC")  # CET end


def test_month_chunks_cover_range_without_gaps():
    chunks = month_chunks(date(2023, 1, 1), date(2026, 10, 1))
    assert len(chunks) == 45
    assert all(prev[1] == nxt[0] for prev, nxt in zip(chunks, chunks[1:]))


def test_partial_months_at_both_ends():
    chunks = month_chunks(date(2025, 9, 15), date(2025, 10, 10))
    assert chunks[0][0] == pd.Timestamp("2025-09-14 22:00", tz="UTC")
    assert chunks[-1][1] == pd.Timestamp("2025-10-09 22:00", tz="UTC")


def test_series_converted_to_utc_and_labelled():
    rows, report = to_long(warsaw_series(date(2026, 1, 15)), "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 24
    assert set(rows["series"]) == {"price_da"}
    assert str(rows["ts_utc"].dt.tz) == "UTC"
    assert rows["ts_utc"].iloc[0] == pd.Timestamp("2026-01-14 23:00", tz="UTC")
    assert set(rows["resolution_min"]) == {60}
    assert (rows["retrieved_at"] == RETRIEVED).all()


def test_rows_outside_chunk_are_trimmed():
    raw = pd.concat([warsaw_series(date(2025, 12, 31)), warsaw_series(date(2026, 1, 1))])
    rows, _ = to_long(raw, "price_da", *JAN_2026, RETRIEVED)
    assert rows["ts_utc"].min() == JAN_2026[0]
    assert len(rows) == 24


def test_dataframe_columns_renamed_and_unknown_slugified():
    idx = delivery_index(date(2026, 1, 15), "15min").tz_convert("Europe/Warsaw")
    raw = pd.DataFrame({"Solar": 1.0, "Wind Onshore": 2.0, "Biomass Other": 3.0}, index=idx)
    rows, _ = to_long(raw, "wind_solar_da_fcst", *JAN_2026, RETRIEVED)
    assert set(rows["series"]) == {"solar_da_fcst", "wind_onshore_da_fcst", "biomass_other_da_fcst"}
    assert set(rows["resolution_min"]) == {15}
    assert len(rows) == 3 * 96


def test_missing_values_dropped_and_gaps_counted():
    raw = warsaw_series(date(2026, 1, 15))
    raw.iloc[5] = np.nan
    chunk = month_chunks(date(2026, 1, 15), date(2026, 1, 16))[0]
    rows, report = to_long(raw, "price_da", *chunk, RETRIEVED)
    assert len(rows) == 23
    assert report["n_missing"] == 4  # one hour = four quarter-hours


def test_finished_month_counts_trailing_gap():
    raw = warsaw_series(date(2026, 1, 15)).iloc[:20]  # last four hours never arrived
    chunk = month_chunks(date(2026, 1, 15), date(2026, 1, 16))[0]
    _, report = to_long(raw, "price_da", *chunk, RETRIEVED)
    assert report["n_missing"] == 16  # four hours


def test_current_month_not_penalised_for_future_hours():
    raw = warsaw_series(date(2026, 10, 6))
    chunk = month_chunks(date(2026, 10, 1), date(2026, 11, 1))[0]
    rows, report = to_long(raw, "price_da", *chunk, RETRIEVED)
    assert len(rows) == 24
    assert report["n_missing"] == 5 * 96  # 1 to 5 October absent, 7 October onward not yet due


def test_exact_duplicates_dropped_and_counted():
    raw = warsaw_series(date(2026, 1, 15))
    raw = pd.concat([raw, raw.iloc[:3]])
    rows, report = to_long(raw, "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 24
    assert report["n_duplicate"] == 3
    assert not report["conflicts"]


def test_conflicting_duplicates_not_stored():
    first = warsaw_series(date(2026, 1, 15), value=100.0)
    second = warsaw_series(date(2026, 1, 15), value=430.0)  # e.g. the same prices in another currency
    rows, report = to_long(pd.concat([first, second]), "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty
    assert report["conflicts"] == ["price_da: 24 intervals with different values"]


def test_series_name_mapping():
    assert series_name("Forecasted Load") == "load_da_fcst"
    assert series_name("Wind Offshore") == "wind_offshore_da_fcst"


def test_resolution_switch_inside_a_month():
    # Hourly until midnight UTC, quarter-hourly after, as the Polish forecasts did on 13 June 2024.
    hourly = delivery_index(date(2024, 6, 12), "h")
    quarter = delivery_index(date(2024, 6, 13), "15min")
    switch = pd.Timestamp("2024-06-13 00:00", tz="UTC")
    idx = hourly.append(quarter[quarter < switch][::4]).append(quarter[quarter >= switch])
    raw = pd.Series(1.0, index=idx.tz_convert("Europe/Warsaw"))
    chunk = month_chunks(date(2024, 6, 12), date(2024, 6, 14))[0]
    rows, report = to_long(raw, "load_da_fcst", *chunk, RETRIEVED)
    by_res = rows.groupby("resolution_min")["ts_utc"].agg(["min", "max", "count"])
    assert by_res.loc[60, "max"] == pd.Timestamp("2024-06-12 23:00", tz="UTC")
    assert by_res.loc[15, "min"] == switch
    assert by_res.loc[60, "count"] == 24 + 2
    assert report["n_missing"] == 0


def test_short_gap_in_quarter_hourly_data():
    idx = delivery_index(date(2026, 1, 15), "15min")
    raw = pd.Series(1.0, index=idx.delete([41, 42, 43]).tz_convert("Europe/Warsaw"))
    chunk = month_chunks(date(2026, 1, 15), date(2026, 1, 16))[0]
    rows, report = to_long(raw, "load_da_fcst", *chunk, RETRIEVED)
    assert set(rows["resolution_min"]) == {15}  # the rows either side of the gap stay 15-minute
    assert report["n_missing"] == 3