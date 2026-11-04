"""Build the model inputs: one row per delivery hour.

Each row holds the forecast time, the hour, weekday and holiday flag, the inputs listed in SPECS, and
the actual price, which models learn from but never see as an input. Every input is checked by
availability.check before the table is returned.

The inputs follow the expert model of Ziel and Weron (2018) and Maciejowska et al. (2022): the price
at the same hour one, two and seven days earlier; the last, lowest and highest price of the day
before; the load forecast; and the wind plus solar forecast, for the day before (the main models) or
for the target day (the papers' setup, which uses a forecast published after 11:00). The carbon
variants add the clearing price of the latest EU emission allowance auction on or before d-2, the
latest public at 11:00 on d-1 (see availability.py), in EUR per tonne of CO2.

"Same hour" means the same hour on the Warsaw clock. When the clocks go forward, the missing 02:00 is
filled with the average of 01:00 and 03:00; when they go back, the two 02:00 values are averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pl_dayahead.availability import FeatureSpec, check
from pl_dayahead.timeutils import calendar_features, delivery_day_bounds, is_holiday, issue_time_utc

SPECS = [
    FeatureSpec("price_lag_1d", "price", -1),  # same hour on the issue day: the latest known prices
    FeatureSpec("price_lag_2d", "price", -2),  # same hour two days before the target day
    FeatureSpec("price_lag_7d", "price", -7),  # same hour one week before the target day
    FeatureSpec("price_last_1d", "price", -1, "last"),
    FeatureSpec("price_min_1d", "price", -1, "min"),
    FeatureSpec("price_max_1d", "price", -1, "max"),
    FeatureSpec("load_fcst", "load_fcst", 0),
    FeatureSpec("renewables_fcst", "renewables_fcst", 0),  # the papers' setup: due at 18:00 on d-1, after the forecast
    FeatureSpec("renewables_fcst_lag_1d", "renewables_fcst", -1),  # the main models: published by 11:00
    FeatureSpec("eua_lag_2d", "eua", -2, "latest"),  # the carbon variants: auctions on d-1 end after 11:00
]
MAX_AUCTION_AGE_DAYS = 31  # the auctions pause for up to four weeks over New Year; an older price means a missing report
PRICE_INPUTS = ["price_lag_1d", "price_lag_2d", "price_lag_7d", "price_last_1d", "price_min_1d", "price_max_1d"]
CALENDAR = ["local_hour", "weekday", "is_holiday"]
TARGET = "price"
FEATURES = CALENDAR + [spec.name for spec in SPECS]
COLUMNS = ["ts_utc", "delivery_day", "issue_time_utc", *FEATURES, TARGET]


def model_inputs(renewables: str) -> list[str]:
    """The inputs of the linear model and the trees, with `renewables` as their wind plus solar column."""
    return [*CALENDAR, *PRICE_INPUTS, "load_fcst", renewables]


def _hours_in_day(day: pd.Timestamp) -> int:
    """Return the number of hours in a Warsaw delivery day: 23, 24 or 25."""
    start, end = delivery_day_bounds(day.date())
    return int((end - start) / pd.Timedelta(hours=1))


def _day_hour_grid(hours: pd.DataFrame, column: str) -> pd.DataFrame:
    """Reshape a column into one row per delivery day and one column per hour, 0 to 23, with the clock-change rules above."""
    grid = hours.pivot_table(index="delivery_day", columns="local_hour", values=column, aggfunc="mean")
    grid = grid.reindex(columns=range(24))
    # Spring clock change, taken from the calendar so that a partial first or last day cannot pass for one:
    # the day has 23 hours and no 02:00. Use the average of 01:00 and 03:00.
    spring = [day for day in grid.index if _hours_in_day(day) == 23]
    grid.loc[spring, 2] = (grid.loc[spring, 1] + grid.loc[spring, 3]) / 2
    return grid


def _whole_day_stat(hours: pd.DataFrame, column: str, agg: str) -> pd.Series:
    """Return the lowest or highest value of a column for each delivery day, left empty if any hour of that day is missing."""
    by_day = hours.groupby("delivery_day")[column]
    values = by_day.min() if agg == "min" else by_day.max()
    expected = pd.Series([_hours_in_day(day) for day in values.index], index=values.index)
    return values.where(by_day.count() == expected)


def _latest_auction_price(auctions: pd.DataFrame | None, days: pd.Series) -> np.ndarray:
    """Return, for each day, the EUA price of the latest auction held on or before it.

    A day with two auctions takes their volume-weighted price. The value is missing if there is no
    auction in the MAX_AUCTION_AGE_DAYS up to the day, or no auctions at all.
    """
    if auctions is None or auctions.empty:
        return np.full(len(days), np.nan)
    auction_day = pd.to_datetime(auctions["auction_date"])
    turnover = (auctions["price"] * auctions["volume"]).groupby(auction_day).sum()
    daily_price = (turnover / auctions["volume"].groupby(auction_day).sum()).sort_index()
    held_on = pd.Series(daily_price.index, index=daily_price.index)
    wanted = daily_price.index.union(pd.DatetimeIndex(days.unique()))
    price = daily_price.reindex(wanted).ffill()
    age = wanted - pd.DatetimeIndex(held_on.reindex(wanted).ffill())
    price[age > pd.Timedelta(days=MAX_AUCTION_AGE_DAYS)] = np.nan
    return days.map(price).to_numpy()


def build_features(hourly: pd.DataFrame, auctions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build the input table from the hourly table, one row per delivery hour.

    `auctions` holds the EUA auctions (auction_date, price, volume) for the carbon input; without
    them that input is missing and only the carbon variants are affected. Returns the table and the
    names of the inputs that rely on the papers' convention. Stops with an error if any input uses
    data not yet published at the forecast time.
    """
    features = hourly.sort_values("ts_utc").reset_index(drop=True)
    calendar = calendar_features(features["ts_utc"])
    features["delivery_day"] = pd.to_datetime(calendar["delivery_day"].to_numpy())  # midnight, no time zone
    features["local_hour"] = calendar["local_hour"].to_numpy()
    features["weekday"] = calendar["weekday"].to_numpy()
    features["is_holiday"] = is_holiday(features["delivery_day"]).to_numpy()
    features["renewables_fcst"] = features["wind_fcst"] + features["solar_fcst"]  # missing if either part is missing
    issue_times = {day: issue_time_utc(day.date()) for day in features["delivery_day"].drop_duplicates()}
    features["issue_time_utc"] = features["delivery_day"].map(issue_times)

    # One day-by-hour grid per series that an input reads from an earlier day, shared by all such inputs.
    lagged = {spec.series for spec in SPECS if spec.agg in ("hour", "last") and spec.day_offset != 0}
    grids = {series: _day_hour_grid(features, series) for series in lagged}

    for spec in SPECS:
        source_day = features["delivery_day"] + pd.Timedelta(days=spec.day_offset)
        if spec.agg == "hour" and spec.day_offset == 0:
            features[spec.name] = features[spec.series]  # the target hour itself
        elif spec.agg == "hour":
            wanted = pd.MultiIndex.from_arrays([source_day, features["local_hour"]])
            features[spec.name] = grids[spec.series].stack().reindex(wanted).to_numpy()
        elif spec.agg == "last":
            features[spec.name] = source_day.map(grids[spec.series][23]).to_numpy()
        elif spec.agg == "latest":
            features[spec.name] = _latest_auction_price(auctions, source_day)
        else:
            features[spec.name] = source_day.map(_whole_day_stat(features, spec.series, spec.agg)).to_numpy()

    return features[COLUMNS], check(SPECS)
