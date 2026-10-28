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
from pl_dayahead.holidays_pl import is_holiday
from pl_dayahead.timeutils import calendar_features, delivery_day_bounds, issue_time_utc

SPECS = [
    FeatureSpec("price_lag_1d", "price", -1),  # same hour on the issue day: the latest known prices
    FeatureSpec("price_lag_2d", "price", -2),  # same hour two days before the target day
    FeatureSpec("price_lag_7d", "price", -7),  # same hour one week before the target day
    FeatureSpec("price_last_1d", "price", -1, "last"),
    FeatureSpec("price_min_1d", "price", -1, "min"),
    FeatureSpec("price_max_1d", "price", -1, "max"),
    FeatureSpec("load_fcst", "load_fcst", 0),
    FeatureSpec("res_fcst", "res_fcst", 0),  # the papers' setup: due at 18:00 on d-1, after the forecast
    FeatureSpec("res_fcst_lag_1d", "res_fcst", -1),  # the main models: published by 11:00
    FeatureSpec("eua_lag_2d", "eua", -2, "latest"),  # the carbon variants: auctions on d-1 end after 11:00
]
MAX_AUCTION_AGE_DAYS = 31  # the auctions pause for up to four weeks over New Year; an older price means a missing report
PRICE_INPUTS = ["price_lag_1d", "price_lag_2d", "price_lag_7d", "price_last_1d", "price_min_1d", "price_max_1d"]
CALENDAR = ["local_hour", "weekday", "is_holiday"]
TARGET = "price"
FEATURES = CALENDAR + [s.name for s in SPECS]
COLUMNS = ["ts_utc", "delivery_day", "issue_time_utc", *FEATURES, TARGET]


def _day_hour_grid(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Reshape a column into one row per delivery day and one column per hour, 0 to 23, with the clock-change rules above."""
    grid = df.pivot_table(index="delivery_day", columns="local_hour", values=column, aggfunc="mean")
    grid = grid.reindex(columns=range(24))
    # Spring clock change: the day has 23 hours and no 02:00. Use the average of 01:00 and 03:00.
    hours_per_day = df.groupby("delivery_day").size()
    spring = hours_per_day.index[hours_per_day == 23].intersection(grid.index)
    grid.loc[spring, 2] = (grid.loc[spring, 1] + grid.loc[spring, 3]) / 2
    return grid


def _hours_in_day(day: pd.Timestamp) -> int:
    """Return the number of hours in a Warsaw delivery day: 23, 24 or 25."""
    start, end = delivery_day_bounds(day.date())
    return int((end - start) / pd.Timedelta(hours=1))


def _whole_day_stat(df: pd.DataFrame, column: str, agg: str) -> pd.Series:
    """Return the lowest or highest value of a column for each delivery day, left empty if any hour of that day is missing."""
    groups = df.groupby("delivery_day")[column]
    stat = groups.min() if agg == "min" else groups.max()
    expected = pd.Series([_hours_in_day(d) for d in stat.index], index=stat.index)
    return stat.where(groups.count() == expected)


def _latest_auction_price(auctions: pd.DataFrame | None, days: pd.Series) -> np.ndarray:
    """Return, for each day, the EUA price of the latest auction held on or before it.

    A day with two auctions takes their volume-weighted price. The value is missing if there is no
    auction in the MAX_AUCTION_AGE_DAYS up to the day, or no auctions at all.
    """
    if auctions is None or auctions.empty:
        return np.full(len(days), np.nan)
    auction_day = pd.to_datetime(auctions["auction_date"])
    turnover = (auctions["price"] * auctions["volume"]).groupby(auction_day).sum()
    daily = (turnover / auctions["volume"].groupby(auction_day).sum()).sort_index()
    held = pd.Series(daily.index, index=daily.index)
    wanted = daily.index.union(pd.DatetimeIndex(days.unique()))
    price = daily.reindex(wanted).ffill()
    age = wanted - pd.DatetimeIndex(held.reindex(wanted).ffill())
    price[age > pd.Timedelta(days=MAX_AUCTION_AGE_DAYS)] = np.nan
    return days.map(price).to_numpy()


def build_features(hourly: pd.DataFrame, auctions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build the input table from the hourly table, one row per delivery hour.

    `auctions` holds the EUA auctions (auction_date, price, volume) for the carbon input; without
    them that input is missing and only the carbon variants are affected. Returns the table and the
    names of the inputs that rely on the papers' convention. Stops with an error if any input uses
    data not yet published at the forecast time.
    """
    df = hourly.sort_values("ts_utc").reset_index(drop=True)
    cal = calendar_features(df["ts_utc"])
    df["delivery_day"] = pd.to_datetime(cal["delivery_day"].to_numpy())  # midnight, no time zone
    df["local_hour"] = cal["local_hour"].to_numpy()
    df["weekday"] = cal["weekday"].to_numpy()
    df["is_holiday"] = is_holiday(df["delivery_day"]).to_numpy()
    df["res_fcst"] = df["wind_fcst"] + df["solar_fcst"]  # missing if either part is missing

    days = df["delivery_day"].drop_duplicates()
    issue = {d: issue_time_utc(d.date()) for d in days}
    df["issue_time_utc"] = df["delivery_day"].map(issue)

    for spec in SPECS:
        source_day = df["delivery_day"] + pd.Timedelta(days=spec.day_offset)
        if spec.agg == "hour" and spec.day_offset == 0:
            df[spec.name] = df[spec.series]  # the target hour itself
        elif spec.agg == "hour":
            lookup = _day_hour_grid(df, spec.series).stack()  # index: (delivery_day, local_hour)
            source = pd.MultiIndex.from_arrays([source_day, df["local_hour"]])
            df[spec.name] = lookup.reindex(source).to_numpy()
        elif spec.agg == "last":
            last_hour = _day_hour_grid(df, spec.series)[23]
            df[spec.name] = source_day.map(last_hour).to_numpy()
        elif spec.agg == "latest":
            df[spec.name] = _latest_auction_price(auctions, source_day)
        else:
            df[spec.name] = source_day.map(_whole_day_stat(df, spec.series, spec.agg)).to_numpy()

    convention = check(SPECS)
    return df[COLUMNS], convention
