"""Offline XML fixtures retain declared interval lengths even when observations are missing."""
from datetime import date
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from pl_dayahead.ingest.entsoe import month_chunks, to_long
from pl_dayahead.staging import build_hourly
from pl_dayahead.timeutils import delivery_day_bounds

RETRIEVED = pd.Timestamp("2026-10-06 12:00", tz="UTC")
JAN_2026 = month_chunks(date(2026, 1, 1), date(2026, 2, 1))[0]


def xml_series(day=date(2026, 1, 15), minutes=15, values=None, curve="A01", psr=None):
    start, end = delivery_day_bounds(day)
    count = int((end - start) / pd.Timedelta(minutes=minutes))
    values = {i: 100.0 for i in range(1, count + 1)} if values is None else values
    series = ET.Element("TimeSeries")
    ET.SubElement(series, "curveType").text = curve
    if psr:
        ET.SubElement(ET.SubElement(series, "MktPSRType"), "psrType").text = psr
    for tag, value in [("quantity_Measure_Unit.name", "MAW"), ("currency_Unit.name", "EUR"),
                       ("price_Measure_Unit.name", "MWH")]:
        ET.SubElement(series, tag).text = value
    period = ET.SubElement(series, "Period")
    interval = ET.SubElement(period, "timeInterval")
    ET.SubElement(interval, "start").text = start.isoformat()
    ET.SubElement(interval, "end").text = end.isoformat()
    ET.SubElement(period, "resolution").text = f"PT{minutes}M"
    for position, value in values.items():
        point = ET.SubElement(period, "Point")
        ET.SubElement(point, "position").text = str(position)
        ET.SubElement(point, "price.amount").text = str(value)
        ET.SubElement(point, "quantity").text = str(value)
    return series


def document(*series):
    root = ET.Element("Publication_MarketDocument", xmlns="urn:entsoe:test")
    root.extend(series)
    return ET.tostring(root, encoding="unicode")


def test_month_chunks_follow_warsaw_midnight_and_cover_range():
    chunks = month_chunks(date(2023, 1, 1), date(2026, 10, 1))
    assert len(chunks) == 45
    assert chunks[0] == (pd.Timestamp("2022-12-31 23:00", tz="UTC"), pd.Timestamp("2023-01-31 23:00", tz="UTC"))
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_month_chunk_across_autumn_dst():
    (start, end), = month_chunks(date(2026, 10, 1), date(2026, 11, 1))
    assert start == pd.Timestamp("2026-09-30 22:00", tz="UTC")
    assert end == pd.Timestamp("2026-10-31 23:00", tz="UTC")


def test_partial_months_at_both_ends():
    chunks = month_chunks(date(2025, 9, 15), date(2025, 10, 10))
    assert chunks[0][0] == pd.Timestamp("2025-09-14 22:00", tz="UTC")
    assert chunks[-1][1] == pd.Timestamp("2025-10-09 22:00", tz="UTC")


def test_source_resolution_and_units_are_preserved():
    rows, conflicts = to_long(document(xml_series(minutes=60)), "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 24 and not conflicts
    assert set(rows.series) == {"price_da"} and set(rows.resolution_min) == {60}
    assert str(rows.ts_utc.dt.tz) == "UTC"
    assert rows.ts_utc.iloc[0] == pd.Timestamp("2026-01-14 23:00", tz="UTC")
    assert (rows.retrieved_at == RETRIEVED).all()


def test_missing_alternate_quarters_never_become_half_hours():
    series = xml_series(values={1: 10, 3: 30, 5: 50, 6: 60, 7: 70, 8: 80})
    rows, conflicts = to_long(document(series), "price_da", *JAN_2026, RETRIEVED)
    assert not conflicts and set(rows.resolution_min) == {15}
    staged, _ = build_hourly(rows)
    assert pd.isna(staged.price.iloc[0]) and staged.price.iloc[1] == 65


def test_single_point_has_known_source_resolution():
    rows, conflicts = to_long(document(xml_series(values={1: 10})), "price_da", *JAN_2026, RETRIEVED)
    assert not conflicts and len(rows) == 1 and rows.resolution_min.iloc[0] == 15
    assert build_hourly(rows)[0].price.isna().all()


def test_a03_step_curve_expands_declared_blocks():
    rows, conflicts = to_long(document(xml_series(curve="A03", values={1: 10, 3: 30})), "price_da", *JAN_2026, RETRIEVED)
    assert not conflicts and len(rows) == 96
    assert rows.value.iloc[:4].tolist() == [10, 10, 30, 30]


def test_step_curve_does_not_fill_across_periods():
    first = xml_series(curve="A03", values={1: 10})
    second = xml_series(day=date(2026, 1, 16), curve="A03", values={3: 30})
    rows, _ = to_long(document(first, second), "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 96 + 94


def test_three_renewables_types_and_load_mapping():
    rows, _ = to_long(document(*(xml_series(psr=p) for p in ["B16", "B18", "B19"])),
                      "wind_solar_da_fcst", *JAN_2026, RETRIEVED)
    assert set(rows.series) == {"solar_da_fcst", "wind_onshore_da_fcst", "wind_offshore_da_fcst"}
    load, _ = to_long(document(xml_series()), "load_da_fcst", *JAN_2026, RETRIEVED)
    assert set(load.series) == {"load_da_fcst"}


def test_exact_duplicates_removed_and_conflicts_rejected():
    rows, conflicts = to_long(document(xml_series(), xml_series()), "price_da", *JAN_2026, RETRIEVED)
    assert len(rows) == 96 and not conflicts
    rows, conflicts = to_long(document(xml_series(), xml_series(values={1: 999})), "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty and "conflicting" in conflicts[0]


def test_conflicting_resolution_rejected_even_when_value_matches():
    rows, conflicts = to_long(document(xml_series(), xml_series(minutes=60)), "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty and conflicts


def test_resolution_change_and_month_trimming():
    start, end = month_chunks(date(2024, 6, 12), date(2024, 6, 14))[0]
    raw = document(xml_series(date(2024, 6, 11), 60), xml_series(date(2024, 6, 12), 60), xml_series(date(2024, 6, 13), 15))
    rows, conflicts = to_long(raw, "load_da_fcst", start, end, RETRIEVED)
    assert not conflicts and rows.ts_utc.min() == start
    assert rows.groupby("resolution_min").size().to_dict() == {15: 96, 60: 24}


@pytest.mark.parametrize("day,count", [(date(2025, 3, 30), 92), (date(2025, 10, 26), 100)])
def test_dst_days_retain_every_source_interval(day, count):
    start, end = delivery_day_bounds(day)
    rows, conflicts = to_long(document(xml_series(day)), "price_da", start, end, RETRIEVED)
    assert not conflicts and len(rows) == count


@pytest.mark.parametrize("minutes,curve", [(5, "A01"), (15, "A02")])
def test_unknown_resolution_or_curve_rejects_series(minutes, curve):
    rows, conflicts = to_long(document(xml_series(minutes=minutes, curve=curve)), "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty and conflicts


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_reject_series(value):
    rows, conflicts = to_long(document(xml_series(values={1: value})), "price_da", *JAN_2026, RETRIEVED)
    assert rows.empty and conflicts


def test_wrong_currency_is_rejected():
    series = xml_series()
    series.find("currency_Unit.name").text = "PLN"
    with pytest.raises(ValueError, match="EUR/MWh"):
        to_long(document(series), "price_da", *JAN_2026, RETRIEVED)
