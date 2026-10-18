"""Build the feature table: one row per delivery hour, with the expert model's inputs.

Each row describes one hour of a target delivery day d and holds:

* when the forecast is made (issue_time_utc: 11:00 Warsaw time on d-1);
* calendar features: local hour, weekday, public holiday;
* data features, each defined by a FeatureSpec: which series, which delivery day relative to
  d, and which value of that day (same hour, last hour, minimum or maximum). Every spec is
  checked by availability.check before the table is returned, so a feature that uses a price
  it could not have known stops the build;
* the target: the actual price of that hour, which models are trained on but never see as input.

The inputs are those of the expert model (Ziel and Weron 2018, Eq. 10; Maciejowska,
Uniejewski and Weron 2022, Eq. 5):

* price_lag_1d, price_lag_2d, price_lag_7d: the price of the same local hour on d-1, d-2, d-7;
* price_last_1d: the price of the last hour of d-1, "the last known price at the time the
  prediction is made" (Maciejowska et al.);
* price_min_1d, price_max_1d: the lowest and highest hourly price of d-1. They link each hour
  to the whole of yesterday, not only to the same hour. They are missing if any hour of d-1
  is missing, so they never rest on part of a day;
* load_fcst: the day-ahead load forecast for the same hour of d;
* res_fcst: the day-ahead renewables forecast (wind plus solar) for the same hour of d, used by
  the main models as in the papers. It is published at 18:00 on d-1, after the issue time;
  see availability.py;
* res_fcst_lag_1d: the renewables forecast for the same hour of d-1, public at 11:00 on d-1,
  used by the comparison models instead.

"Same hour" means the same local wall-clock hour. Two clock-change cases need a rule:
* the spring day has no 02:00, so when that day is a lag source its 02:00 value is the average
  of 01:00 and 03:00;
* the autumn day has 02:00 twice, so as a lag source its two 02:00 values are averaged, and as
  a target day both 02:00 rows get the same lag values.
The minimum and maximum use the actual hours of the source day: 23, 24 or 25 of them.
"""

from __future__ import annotations

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
    FeatureSpec("res_fcst", "res_fcst", 0),  # main models: the papers' convention
    FeatureSpec("res_fcst_lag_1d", "res_fcst", -1),  # comparison models: public at 11:00
]
PRICE_INPUTS = ["price_lag_1d", "price_lag_2d", "price_lag_7d", "price_last_1d", "price_min_1d", "price_max_1d"]
CALENDAR = ["local_hour", "weekday", "is_holiday"]
TARGET = "price"
FEATURES = CALENDAR + [s.name for s in SPECS]
COLUMNS = ["ts_utc", "delivery_day", "issue_time_utc", *FEATURES, TARGET]


def _day_hour_grid(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Values of `column` with one row per delivery day and one column per local hour (0 to 23)."""
    grid = df.pivot_table(index="delivery_day", columns="local_hour", values=column, aggfunc="mean")
    grid = grid.reindex(columns=range(24))
    # Spring clock change: the day has 23 hours and no 02:00. Use the average of 01:00 and 03:00.
    hours_per_day = df.groupby("delivery_day").size()
    spring = hours_per_day.index[hours_per_day == 23].intersection(grid.index)
    grid.loc[spring, 2] = (grid.loc[spring, 1] + grid.loc[spring, 3]) / 2
    return grid


def _hours_in_day(day: pd.Timestamp) -> int:
    start, end = delivery_day_bounds(day.date())
    return int((end - start) / pd.Timedelta(hours=1))


def _whole_day_stat(df: pd.DataFrame, column: str, agg: str) -> pd.Series:
    """min or max of `column` per delivery day, missing unless every hour of the day has a value."""
    groups = df.groupby("delivery_day")[column]
    stat = groups.min() if agg == "min" else groups.max()
    expected = pd.Series([_hours_in_day(d) for d in stat.index], index=stat.index)
    return stat.where(groups.count() == expected)


def build_features(hourly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Feature table from the staged hourly table.

    Returns the table and the names of the features that rely on the papers' convention
    (see availability.py).
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
        else:
            df[spec.name] = source_day.map(_whole_day_stat(df, spec.series, spec.agg)).to_numpy()

    convention = check(SPECS, [d.date() for d in days])
    return df[COLUMNS], convention
