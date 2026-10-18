from datetime import date, timedelta

import numpy as np
import pandas as pd

from pl_dayahead.features import COLUMNS, build_features
from pl_dayahead.models.boosting import GradientBoostingResD1
from pl_dayahead.models.linear import LinearResD1
from pl_dayahead.timeutils import LOCAL_TZ, delivery_index


def staged(first: date, n_days: int) -> pd.DataFrame:
    """A synthetic hourly table, each value unique so lags can be traced back."""
    idx = delivery_index(first, "h")
    for i in range(1, n_days):
        idx = idx.append(delivery_index(first + timedelta(days=i), "h"))
    n = len(idx)
    base = np.arange(n, dtype=float)
    return pd.DataFrame({
        "ts_utc": idx, "price": 1000 + base, "load_fcst": 20000 + base, "solar_fcst": 3000 + base,
        "wind_onshore_fcst": 4000 + base, "wind_offshore_fcst": 0.0, "wind_fcst": 4000 + base,
        "residual_load_fcst": 13000 - base,
    })


def price_at(hourly: pd.DataFrame, day: date, hour: int) -> pd.Series:
    local = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ)
    return hourly.loc[(local.dt.date == day) & (local.dt.hour == hour), "price"]


def rows(feats: pd.DataFrame, day: date, hour: int | None = None) -> pd.DataFrame:
    sel = feats["delivery_day"] == pd.Timestamp(day)
    if hour is not None:
        sel &= feats["local_hour"] == hour
    return feats[sel]


def set_day(hourly: pd.DataFrame, day: date, column: str, values) -> pd.DataFrame:
    """Overwrite `column` for every hour of a local delivery day, in time order."""
    out = hourly.copy()
    on_day = (out["ts_utc"].dt.tz_convert(LOCAL_TZ).dt.date == day).to_numpy()
    out.loc[on_day, column] = values
    return out


def test_columns_and_one_row_per_hour():
    hourly = staged(date(2026, 1, 1), 10)
    feats, convention = build_features(hourly)
    assert list(feats.columns) == COLUMNS
    assert len(feats) == len(hourly)
    assert convention == ["res_fcst"]  # only the day-d renewables forecast is published after 11:00


def test_price_lags_use_the_same_local_hour():
    hourly = staged(date(2026, 1, 1), 10)
    feats, _ = build_features(hourly)
    target = rows(feats, date(2026, 1, 9), 18)
    assert target["price_lag_1d"].item() == price_at(hourly, date(2026, 1, 8), 18).item()
    assert target["price_lag_2d"].item() == price_at(hourly, date(2026, 1, 7), 18).item()
    assert target["price_lag_7d"].item() == price_at(hourly, date(2026, 1, 2), 18).item()


def test_first_week_has_no_weekly_lag():
    feats, _ = build_features(staged(date(2026, 1, 1), 10))
    assert rows(feats, date(2026, 1, 7))["price_lag_7d"].isna().all()
    assert rows(feats, date(2026, 1, 8))["price_lag_7d"].notna().all()


def test_issue_time_and_holiday():
    feats, _ = build_features(staged(date(2025, 12, 30), 10))
    jan1 = rows(feats, date(2026, 1, 1))
    assert (jan1["issue_time_utc"] == pd.Timestamp("2025-12-31 10:00", tz="UTC")).all()
    assert jan1["is_holiday"].all()
    assert not rows(feats, date(2026, 1, 2))["is_holiday"].any()


def test_spring_clock_change_lag_is_interpolated():
    hourly = staged(date(2026, 3, 27), 5)  # 29 March has no 02:00
    feats, _ = build_features(hourly)
    lag = rows(feats, date(2026, 3, 30), 2)["price_lag_1d"].item()
    expected = (price_at(hourly, date(2026, 3, 29), 1).item() + price_at(hourly, date(2026, 3, 29), 3).item()) / 2
    assert lag == expected


def test_autumn_clock_change_both_ways():
    hourly = staged(date(2025, 10, 24), 5)  # 26 October has 02:00 twice
    feats, _ = build_features(hourly)
    target_day = rows(feats, date(2025, 10, 26), 2)
    assert len(target_day) == 2
    assert target_day["price_lag_1d"].nunique() == 1  # both 02:00 rows get 25 October's 02:00
    after = rows(feats, date(2025, 10, 27), 2)["price_lag_1d"].item()
    assert after == price_at(hourly, date(2025, 10, 26), 2).mean()


def test_last_price_minimum_and_maximum_of_the_previous_day():
    hourly = staged(date(2026, 1, 1), 10)
    prices = np.full(24, 100.0)
    prices[5], prices[18], prices[23] = 20.0, 300.0, 150.0  # minimum at 05:00, maximum at 18:00
    hourly = set_day(hourly, date(2026, 1, 8), "price", prices)
    feats, _ = build_features(hourly)
    for hour in [0, 12, 23]:  # the same three values in every hour of the target day
        target = rows(feats, date(2026, 1, 9), hour)
        assert target["price_min_1d"].item() == 20.0
        assert target["price_max_1d"].item() == 300.0
        assert target["price_last_1d"].item() == 150.0


def test_minimum_and_maximum_use_all_25_hours_of_the_autumn_day():
    hourly = staged(date(2025, 10, 24), 5)  # 26 October has 02:00 twice
    prices = np.full(25, 100.0)
    prices[3] = 5.0  # the second 02:00
    hourly = set_day(hourly, date(2025, 10, 26), "price", prices)
    feats, _ = build_features(hourly)
    assert rows(feats, date(2025, 10, 27), 9)["price_min_1d"].item() == 5.0
    assert rows(feats, date(2025, 10, 27), 9)["price_max_1d"].item() == 100.0


def test_minimum_and_maximum_are_missing_if_an_hour_of_the_day_is_missing():
    hourly = staged(date(2026, 1, 1), 10)
    prices = np.arange(24, dtype=float)
    prices[7] = np.nan
    hourly = set_day(hourly, date(2026, 1, 8), "price", prices)
    target = rows(build_features(hourly)[0], date(2026, 1, 9))
    assert target["price_min_1d"].isna().all() and target["price_max_1d"].isna().all()
    assert (target["price_last_1d"] == 23.0).all()  # the last hour itself is there


def test_renewables_are_wind_plus_solar_for_day_d_and_for_the_day_before():
    hourly = staged(date(2026, 1, 1), 10)
    feats, _ = build_features(hourly)
    local = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ)

    def res(day, hour):
        at = (local.dt.date == day) & (local.dt.hour == hour)
        return (hourly.loc[at, "wind_fcst"] + hourly.loc[at, "solar_fcst"]).item()

    target = rows(feats, date(2026, 1, 9), 13)
    assert target["res_fcst"].item() == res(date(2026, 1, 9), 13)
    assert target["res_fcst_lag_1d"].item() == res(date(2026, 1, 8), 13)


def truncate(hourly: pd.DataFrame, target: date, renewables_of_target_day: bool) -> pd.DataFrame:
    """Delete what was not public at 11:00 on the day before `target`."""
    local_day = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    cut = hourly.copy()
    cut.loc[local_day >= target, "price"] = np.nan  # target-day prices come after the 12:00 auction
    cut.loc[local_day > target, ["load_fcst", "solar_fcst", "wind_fcst", "residual_load_fcst"]] = np.nan
    if renewables_of_target_day:
        cut.loc[local_day == target, ["solar_fcst", "wind_fcst"]] = np.nan  # due at 18:00 on the day before
    return cut


def test_no_price_of_the_target_day_or_later_is_used():
    """Truncation test 1: delete every price from the target day on; no feature may change."""
    hourly = staged(date(2026, 1, 1), 20)
    target = date(2026, 1, 12)
    full, _ = build_features(hourly)
    truncated, _ = build_features(truncate(hourly, target, renewables_of_target_day=False))
    cols = [c for c in COLUMNS if c != "price"]
    pd.testing.assert_frame_equal(
        rows(full, target)[cols].reset_index(drop=True),
        rows(truncated, target)[cols].reset_index(drop=True),
    )


def test_comparison_inputs_were_all_public_at_the_issue_time():
    """Truncation test 2: also delete the target day's wind and solar forecasts.

    The comparison models' inputs must not change. The main models' renewables input does
    change, which shows the test can see a late input when there is one.
    """
    hourly = staged(date(2026, 1, 1), 20)
    target = date(2026, 1, 12)
    full, _ = build_features(hourly)
    truncated, _ = build_features(truncate(hourly, target, renewables_of_target_day=True))
    cols = sorted(set(LinearResD1.features) | set(GradientBoostingResD1.features))
    assert "res_fcst" not in cols
    pd.testing.assert_frame_equal(
        rows(full, target)[cols].reset_index(drop=True),
        rows(truncated, target)[cols].reset_index(drop=True),
    )
    assert rows(truncated, target)["res_fcst"].isna().all()
    assert rows(full, target)["res_fcst"].notna().all()
