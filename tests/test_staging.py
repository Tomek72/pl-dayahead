"""Tests for staging. Pure pandas, no database needed."""

from datetime import date

import numpy as np
import pandas as pd

from pl_dayahead.staging import build_hourly
from pl_dayahead.timeutils import delivery_index


def raw_rows(series: str, idx: pd.DatetimeIndex, values, resolution: int) -> pd.DataFrame:
    """Rows as they come out of entsoe_latest."""
    return pd.DataFrame(
        {"series": series, "ts_utc": idx, "resolution_min": resolution, "value": values}
    )


def full_day(day: date, freq: str = "h", **values) -> pd.DataFrame:
    """All five series for one delivery day, each at a constant value."""
    idx = delivery_index(day, freq)
    res = 60 if freq == "h" else 15
    defaults = dict(price_da=100.0, load_da_fcst=20000.0, solar_da_fcst=3000.0,
                    wind_onshore_da_fcst=5000.0, wind_offshore_da_fcst=0.0)
    defaults.update(values)
    return pd.concat(raw_rows(s, idx, v, res) for s, v in defaults.items())


def test_hourly_rows_pass_through():
    hourly, report = build_hourly(full_day(date(2024, 1, 15)))
    assert len(hourly) == 24
    assert (hourly["price"] == 100.0).all()
    assert report["missing"]["price"] == 0


def test_quarter_hours_averaged_to_the_hour():
    idx = delivery_index(date(2026, 1, 15), "15min")
    prices = np.tile([80.0, 90.0, 110.0, 120.0], 24)
    raw = pd.concat([full_day(date(2026, 1, 15), "15min").query("series != 'price_da'"),
                     raw_rows("price_da", idx, prices, 15)])
    hourly, _ = build_hourly(raw)
    assert len(hourly) == 24
    assert np.allclose(hourly["price"], 100.0)


def test_partly_covered_hour_becomes_missing():
    raw = full_day(date(2026, 1, 15), "15min")
    drop = (raw["series"] == "price_da") & (raw["ts_utc"] == pd.Timestamp("2026-01-15 09:15", tz="UTC"))
    hourly, report = build_hourly(raw[~drop])
    assert hourly.loc[hourly["ts_utc"] == pd.Timestamp("2026-01-15 09:00", tz="UTC"), "price"].isna().all()
    assert report["missing"]["price"] == 1
    assert report["dropped_hours"] == {"price_da": 1}


def test_resolution_switch_inside_a_day():
    day = date(2024, 6, 13)
    hourly_part = full_day(day, "h")
    quarter_part = full_day(day, "15min")
    switch = pd.Timestamp("2024-06-13 00:00", tz="UTC")
    raw = pd.concat([hourly_part[hourly_part["ts_utc"] < switch],
                     quarter_part[quarter_part["ts_utc"] >= switch]])
    hourly, report = build_hourly(raw)
    assert len(hourly) == 24
    assert all(n == 0 for n in report["missing"].values())


def test_missing_hour_appears_as_a_row():
    raw = full_day(date(2024, 1, 15))
    gap = pd.Timestamp("2024-01-15 10:00", tz="UTC")
    hourly, report = build_hourly(raw[~((raw["series"] == "load_da_fcst") & (raw["ts_utc"] == gap))])
    assert len(hourly) == 24
    assert report["missing"]["load_fcst"] == 1
    assert report["missing"]["residual_load_fcst"] == 1  # residual load needs load


def test_offshore_zero_before_its_first_forecast_and_missing_after():
    raw = pd.concat([full_day(date(2026, 6, 16)).query("series != 'wind_offshore_da_fcst'"),
                     full_day(date(2026, 6, 17), wind_offshore_da_fcst=800.0)])
    later = pd.Timestamp("2026-06-17 05:00", tz="UTC")
    raw = raw[~((raw["series"] == "wind_offshore_da_fcst") & (raw["ts_utc"] == later))]
    hourly, report = build_hourly(raw)
    first_day = hourly["ts_utc"] < pd.Timestamp("2026-06-16 22:00", tz="UTC")
    assert (hourly.loc[first_day, "wind_fcst"] == 5000.0).all()  # onshore only
    assert hourly.loc[hourly["ts_utc"] == later, "wind_fcst"].isna().all()
    assert report["missing"]["wind_fcst"] == 1


def test_residual_load_is_load_minus_wind_minus_solar():
    hourly, _ = build_hourly(full_day(date(2026, 6, 20), "15min", wind_offshore_da_fcst=500.0))
    assert np.allclose(hourly["residual_load_fcst"], 20000.0 - 5000.0 - 500.0 - 3000.0)


def test_autumn_clock_change_has_25_hours():
    hourly, _ = build_hourly(full_day(date(2025, 10, 26), "15min"))
    assert len(hourly) == 25
    assert str(hourly["ts_utc"].dt.tz) == "UTC"


def test_unknown_series_reported_not_used():
    raw = pd.concat([full_day(date(2024, 1, 15)),
                     raw_rows("biomass_da_fcst", delivery_index(date(2024, 1, 15)), 1.0, 60)])
    hourly, report = build_hourly(raw)
    assert report["ignored_series"] == ["biomass_da_fcst"]
    assert "biomass_da_fcst" not in hourly.columns