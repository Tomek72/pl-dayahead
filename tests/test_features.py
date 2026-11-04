from datetime import date, timedelta

import numpy as np
import pandas as pd

from pl_dayahead.features import COLUMNS, build_features
from pl_dayahead.models import MODELS, PAPERS_SETUP
from pl_dayahead.timeutils import LOCAL_TZ, delivery_index


def hourly_table(first: date, n_days: int) -> pd.DataFrame:
    """A synthetic hourly table, each value unique so lags can be traced back."""
    hours = delivery_index(first, "h")
    for offset in range(1, n_days):
        hours = hours.append(delivery_index(first + timedelta(days=offset), "h"))
    step = np.arange(len(hours), dtype=float)
    return pd.DataFrame({
        "ts_utc": hours, "price": 1000 + step, "load_fcst": 20000 + step, "solar_fcst": 3000 + step,
        "wind_onshore_fcst": 4000 + step, "wind_offshore_fcst": 0.0, "wind_fcst": 4000 + step,
        "residual_load_fcst": 13000 - step,
    })


def price_at(hourly: pd.DataFrame, day: date, hour: int) -> pd.Series:
    local = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ)
    return hourly.loc[(local.dt.date == day) & (local.dt.hour == hour), "price"]


def feature_rows(features: pd.DataFrame, day: date, hour: int | None = None) -> pd.DataFrame:
    selected = features["delivery_day"] == pd.Timestamp(day)
    if hour is not None:
        selected &= features["local_hour"] == hour
    return features[selected]


def set_day(hourly: pd.DataFrame, day: date, column: str, values) -> pd.DataFrame:
    """Overwrite `column` for every hour of a local delivery day, in time order."""
    changed = hourly.copy()
    on_day = (changed["ts_utc"].dt.tz_convert(LOCAL_TZ).dt.date == day).to_numpy()
    changed.loc[on_day, column] = values
    return changed


def auctions(*day_price: tuple[date, float]) -> pd.DataFrame:
    """EUA auctions, one per (day, price) pair, each of the same volume."""
    return pd.DataFrame({"auction_date": [day for day, _ in day_price], "price": [price for _, price in day_price],
                         "volume": 1.0})


def carbon(features: pd.DataFrame, day: date) -> float:
    """The carbon input of a delivery day, the same for each of its hours."""
    values = feature_rows(features, day)["eua_lag_2d"].unique()
    assert len(values) == 1
    return values[0]


def test_columns_and_one_row_per_hour():
    hourly = hourly_table(date(2026, 1, 1), 10)
    features, convention = build_features(hourly)
    assert list(features.columns) == COLUMNS
    assert len(features) == len(hourly)
    assert convention == ["renewables_fcst"]  # only the day-d renewables forecast is published after 11:00


def test_price_lags_use_the_same_local_hour():
    hourly = hourly_table(date(2026, 1, 1), 10)
    target = feature_rows(build_features(hourly)[0], date(2026, 1, 9), 18)
    assert target["price_lag_1d"].item() == price_at(hourly, date(2026, 1, 8), 18).item()
    assert target["price_lag_2d"].item() == price_at(hourly, date(2026, 1, 7), 18).item()
    assert target["price_lag_7d"].item() == price_at(hourly, date(2026, 1, 2), 18).item()


def test_first_week_has_no_weekly_lag():
    features, _ = build_features(hourly_table(date(2026, 1, 1), 10))
    assert feature_rows(features, date(2026, 1, 7))["price_lag_7d"].isna().all()
    assert feature_rows(features, date(2026, 1, 8))["price_lag_7d"].notna().all()


def test_issue_time_and_holiday():
    features, _ = build_features(hourly_table(date(2025, 12, 30), 10))
    new_year = feature_rows(features, date(2026, 1, 1))
    assert (new_year["issue_time_utc"] == pd.Timestamp("2025-12-31 10:00", tz="UTC")).all()
    assert new_year["is_holiday"].all()
    assert not feature_rows(features, date(2026, 1, 2))["is_holiday"].any()


def test_spring_clock_change_lag_is_interpolated():
    hourly = hourly_table(date(2026, 3, 27), 5)  # 29 March has no 02:00
    lag = feature_rows(build_features(hourly)[0], date(2026, 3, 30), 2)["price_lag_1d"].item()
    expected = (price_at(hourly, date(2026, 3, 29), 1).item() + price_at(hourly, date(2026, 3, 29), 3).item()) / 2
    assert lag == expected


def test_a_partial_first_day_is_not_taken_for_the_spring_clock_change():
    hourly = hourly_table(date(2026, 1, 1), 10).iloc[1:].copy()  # starts at 01:00, so 1 January has 23 rows
    local = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ)
    hourly.loc[(local.dt.date == date(2026, 1, 1)) & (local.dt.hour == 2), "price"] = 5000.0  # unlike 01:00 and 03:00
    lag = feature_rows(build_features(hourly)[0], date(2026, 1, 2), 2)["price_lag_1d"].item()
    assert lag == 5000.0  # the real 02:00, not the mean of 01:00 and 03:00


def test_autumn_clock_change_both_ways():
    hourly = hourly_table(date(2025, 10, 24), 5)  # 26 October has 02:00 twice
    features, _ = build_features(hourly)
    target_day = feature_rows(features, date(2025, 10, 26), 2)
    assert len(target_day) == 2
    assert target_day["price_lag_1d"].nunique() == 1  # both 02:00 rows get 25 October's 02:00
    after = feature_rows(features, date(2025, 10, 27), 2)["price_lag_1d"].item()
    assert after == price_at(hourly, date(2025, 10, 26), 2).mean()


def test_last_price_minimum_and_maximum_of_the_previous_day():
    prices = np.full(24, 100.0)
    prices[5], prices[18], prices[23] = 20.0, 300.0, 150.0  # minimum at 05:00, maximum at 18:00
    hourly = set_day(hourly_table(date(2026, 1, 1), 10), date(2026, 1, 8), "price", prices)
    features, _ = build_features(hourly)
    for hour in [0, 12, 23]:  # the same three values in every hour of the target day
        target = feature_rows(features, date(2026, 1, 9), hour)
        assert target["price_min_1d"].item() == 20.0
        assert target["price_max_1d"].item() == 300.0
        assert target["price_last_1d"].item() == 150.0


def test_minimum_and_maximum_use_all_25_hours_of_the_autumn_day():
    prices = np.full(25, 100.0)
    prices[3] = 5.0  # the second 02:00
    hourly = set_day(hourly_table(date(2025, 10, 24), 5), date(2025, 10, 26), "price", prices)  # 26 October has 02:00 twice
    target = feature_rows(build_features(hourly)[0], date(2025, 10, 27), 9)
    assert target["price_min_1d"].item() == 5.0
    assert target["price_max_1d"].item() == 100.0


def test_minimum_and_maximum_are_missing_if_an_hour_of_the_day_is_missing():
    prices = np.arange(24, dtype=float)
    prices[7] = np.nan
    hourly = set_day(hourly_table(date(2026, 1, 1), 10), date(2026, 1, 8), "price", prices)
    target = feature_rows(build_features(hourly)[0], date(2026, 1, 9))
    assert target["price_min_1d"].isna().all() and target["price_max_1d"].isna().all()
    assert (target["price_last_1d"] == 23.0).all()  # the last hour itself is there


def test_renewables_are_wind_plus_solar_for_day_d_and_for_the_day_before():
    hourly = hourly_table(date(2026, 1, 1), 10)
    local = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ)

    def renewables(day, hour):
        selected = (local.dt.date == day) & (local.dt.hour == hour)
        return (hourly.loc[selected, "wind_fcst"] + hourly.loc[selected, "solar_fcst"]).item()

    target = feature_rows(build_features(hourly)[0], date(2026, 1, 9), 13)
    assert target["renewables_fcst"].item() == renewables(date(2026, 1, 9), 13)
    assert target["renewables_fcst_lag_1d"].item() == renewables(date(2026, 1, 8), 13)


def truncate(hourly: pd.DataFrame, target: date, renewables_of_target_day: bool) -> pd.DataFrame:
    """Delete what was not public at 11:00 on the day before `target`."""
    local_day = hourly["ts_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    truncated = hourly.copy()
    truncated.loc[local_day >= target, "price"] = np.nan  # target-day prices come after the 12:00 auction
    truncated.loc[local_day > target, ["load_fcst", "solar_fcst", "wind_fcst", "residual_load_fcst"]] = np.nan
    if renewables_of_target_day:
        truncated.loc[local_day == target, ["solar_fcst", "wind_fcst"]] = np.nan  # due at 18:00 on the day before
    return truncated


def test_no_price_of_the_target_day_or_later_is_used():
    """Truncation test 1: delete every price from the target day on; no feature may change."""
    hourly = hourly_table(date(2026, 1, 1), 20)
    target = date(2026, 1, 12)
    full, _ = build_features(hourly)
    truncated, _ = build_features(truncate(hourly, target, renewables_of_target_day=False))
    inputs = [column for column in COLUMNS if column != "price"]
    pd.testing.assert_frame_equal(
        feature_rows(full, target)[inputs].reset_index(drop=True),
        feature_rows(truncated, target)[inputs].reset_index(drop=True),
    )


def test_every_input_outside_the_papers_setup_was_public_at_the_issue_time():
    """Truncation test 2: also delete the target day's wind and solar forecasts.

    No input of any model outside the papers' setup may change. The papers' renewables input
    does change, which shows the test can see a late input when there is one.
    """
    hourly = hourly_table(date(2026, 1, 1), 20)
    target = date(2026, 1, 12)
    full, _ = build_features(hourly)
    truncated, _ = build_features(truncate(hourly, target, renewables_of_target_day=True))
    inputs = sorted({column for name, model in MODELS.items() if name not in PAPERS_SETUP.values()
                     for column in model.features})
    assert "renewables_fcst" not in inputs
    pd.testing.assert_frame_equal(
        feature_rows(full, target)[inputs].reset_index(drop=True),
        feature_rows(truncated, target)[inputs].reset_index(drop=True),
    )
    assert feature_rows(truncated, target)["renewables_fcst"].isna().all()
    assert feature_rows(full, target)["renewables_fcst"].notna().all()


def test_carbon_price_is_the_latest_auction_on_or_before_two_days_earlier():
    hourly = hourly_table(date(2026, 1, 1), 12)
    features, _ = build_features(hourly, auctions((date(2025, 12, 29), 70.0), (date(2026, 1, 5), 80.0),
                                                  (date(2026, 1, 8), 82.0)))
    assert carbon(features, date(2026, 1, 10)) == 82.0  # d-2 is 8 January, an auction day
    assert carbon(features, date(2026, 1, 9)) == 80.0  # d-2 is 7 January; the latest auction was on the 5th
    assert carbon(features, date(2026, 1, 2)) == 70.0  # over New Year the last auction's price carries on
    assert build_features(hourly)[0]["eua_lag_2d"].isna().all()  # without auctions the input is missing


def test_a_day_with_two_auctions_takes_their_volume_weighted_price():
    two_auctions = pd.DataFrame({"auction_date": [date(2026, 1, 5)] * 2, "price": [80.0, 86.0], "volume": [3.0, 1.0]})
    features, _ = build_features(hourly_table(date(2026, 1, 1), 10), two_auctions)
    assert carbon(features, date(2026, 1, 7)) == 81.5  # (3 x 80 + 1 x 86) / 4


def test_an_auction_more_than_a_month_old_is_not_used():
    features, _ = build_features(hourly_table(date(2026, 1, 1), 50), auctions((date(2026, 1, 1), 80.0)))
    assert carbon(features, date(2026, 2, 3)) == 80.0  # d-2 is 1 February, 31 days after the auction
    assert np.isnan(carbon(features, date(2026, 2, 4)))  # 32 days: a report must be missing


def test_no_auction_from_the_issue_day_on_is_used():
    """Truncation test 3: delete every auction from d-1 on; the carbon input for day d may not change.

    Deleting the auction on d-2 as well does change it, which shows the test can see a change.
    """
    hourly = hourly_table(date(2026, 1, 1), 20)
    target = date(2026, 1, 15)
    daily_auctions = auctions(*[(date(2026, 1, 1) + timedelta(days=offset), 70.0 + offset) for offset in range(20)])
    full, _ = build_features(hourly, daily_auctions)
    cut, _ = build_features(hourly, daily_auctions[daily_auctions["auction_date"] < target - timedelta(days=1)])
    assert carbon(full, target) == carbon(cut, target) == 82.0  # the auction on 13 January, d-2
    earlier, _ = build_features(hourly, daily_auctions[daily_auctions["auction_date"] < target - timedelta(days=2)])
    assert carbon(earlier, target) == 81.0
