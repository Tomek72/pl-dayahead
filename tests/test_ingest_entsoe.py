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
    return pd.Series(value, index=delivery_index(day, freq).tz_convert("Europe/Warsaw"))


def test_month_chunks_follow_warsaw_midnight():
    (start, end), = month_chunks(date(2023, 1, 1), date(2023, 2, 1))
    assert start == pd.Timestamp("2022-12-31 23:00", tz="UTC")
    assert end == pd.Timestamp("2023-01-31 23:00", tz="UTC")


def test_month_chunk_across_autumn_dst():
    (start, end), = month_chunks(date(2026, 10, 1), date(2026, 11, 1))
    assert start == pd.Timestamp("2026-09-30 22:00", tz="UTC")  # CEST start
    assert end == pd.Timestamp("2026-10-31 23:00", tz="UTC")  # CET end


def test_month_chunks_cover_range_without_gaps():
    chunks = month_chunks(date(2023, 1, 1), date(2026, 10, 1))
    assert len(chunks) == 45
    assert all(earlier[1] == later[0] for earlier, later in zip(chunks, chunks[1:]))


def test_partial_months_at_both_ends():
    chunks = month_chunks(date(2025, 9, 15), date(2025, 10, 10))
    assert chunks[0][0] == pd.Timestamp("2025-09-14 22:00", tz="UTC")
    assert chunks[-1][1] == pd.Timestamp("2025-10-09 22:00", tz="UTC")


def test_series_converted_to_utc_and_labelled():
    rows, conflicts = to_long(warsaw_series(date(2026, 1, 15)), "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 24
    assert set(rows["series"]) == {"price_da"}
    assert str(rows["ts_utc"].dt.tz) == "UTC"
    assert rows["ts_utc"].iloc[0] == pd.Timestamp("2026-01-14 23:00", tz="UTC")
    assert set(rows["resolution_min"]) == {60}
    assert (rows["retrieved_at"] == RETRIEVED).all()
    assert conflicts == []


def test_rows_outside_chunk_are_trimmed():
    raw = pd.concat([warsaw_series(date(2025, 12, 31)), warsaw_series(date(2026, 1, 1))])
    rows, _ = to_long(raw, "price_da", *JAN_2026, RETRIEVED)
    assert rows["ts_utc"].min() == JAN_2026[0]
    assert len(rows) == 24


def test_dataframe_columns_renamed_and_unknown_slugified():
    quarter_hours = delivery_index(date(2026, 1, 15), "15min").tz_convert("Europe/Warsaw")
    raw = pd.DataFrame({"Solar": 1.0, "Wind Onshore": 2.0, "Biomass Other": 3.0}, index=quarter_hours)
    rows, _ = to_long(raw, "wind_solar_da_fcst", *JAN_2026, RETRIEVED)
    assert set(rows["series"]) == {"solar_da_fcst", "wind_onshore_da_fcst", "biomass_other_da_fcst"}
    assert set(rows["resolution_min"]) == {15}
    assert len(rows) == 3 * 96


def test_missing_values_dropped():
    raw = warsaw_series(date(2026, 1, 15))
    raw.iloc[5] = np.nan
    rows, _ = to_long(raw, "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 23


def test_exact_duplicates_dropped():
    raw = warsaw_series(date(2026, 1, 15))
    raw = pd.concat([raw, raw.iloc[:3]])
    rows, conflicts = to_long(raw, "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 24
    assert conflicts == []


def test_conflicting_duplicates_not_stored():
    first = warsaw_series(date(2026, 1, 15), value=100.0)
    second = warsaw_series(date(2026, 1, 15), value=430.0)  # e.g. the same prices in another currency
    rows, conflicts = to_long(pd.concat([first, second]), "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty
    assert conflicts == ["price_da: 24 timestamps with more than one value"]


def test_a_single_row_is_not_stored():
    raw = warsaw_series(date(2026, 1, 15), "15min").iloc[:1]  # one quarter-hour, with no neighbour to measure from
    rows, conflicts = to_long(raw, "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty
    assert conflicts == ["price_da: no two rows are 15, 30 or 60 minutes apart, so the minutes each covers are unknown"]


def test_series_name_mapping():
    assert series_name("Forecasted Load") == "load_da_fcst"
    assert series_name("Wind Offshore") == "wind_offshore_da_fcst"


def test_resolution_switch_inside_a_month():
    # Hourly until midnight UTC, quarter-hourly after, as the Polish forecasts did on 13 June 2024.
    hourly = delivery_index(date(2024, 6, 12), "h")
    quarter = delivery_index(date(2024, 6, 13), "15min")
    switch = pd.Timestamp("2024-06-13 00:00", tz="UTC")
    switching = hourly.append(quarter[quarter < switch][::4]).append(quarter[quarter >= switch])
    raw = pd.Series(1.0, index=switching.tz_convert("Europe/Warsaw"))
    chunk = month_chunks(date(2024, 6, 12), date(2024, 6, 14))[0]
    rows, _ = to_long(raw, "load_da_fcst", *chunk, RETRIEVED)
    by_resolution = rows.groupby("resolution_min")["ts_utc"].agg(["min", "max", "count"])
    assert by_resolution.loc[60, "max"] == pd.Timestamp("2024-06-12 23:00", tz="UTC")
    assert by_resolution.loc[15, "min"] == switch
    assert by_resolution.loc[60, "count"] == 24 + 2


def test_short_gap_in_quarter_hourly_data():
    quarter_hours = delivery_index(date(2026, 1, 15), "15min")
    raw = pd.Series(1.0, index=quarter_hours.delete([41, 42, 43]).tz_convert("Europe/Warsaw"))
    chunk = month_chunks(date(2026, 1, 15), date(2026, 1, 16))[0]
    rows, _ = to_long(raw, "load_da_fcst", *chunk, RETRIEVED)
    assert set(rows["resolution_min"]) == {15}  # the rows either side of the gap stay 15-minute
    assert len(rows) == 96 - 3
