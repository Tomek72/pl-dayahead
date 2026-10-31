"""Test data shared by several test files."""

import numpy as np
import pandas as pd

from pl_dayahead.timeutils import delivery_day_bounds


def hours_between(first: str, last: str) -> pd.DatetimeIndex:
    """The UTC start of every hour of the delivery days from `first` to `last`."""
    start = delivery_day_bounds(pd.Timestamp(first).date())[0]
    end = delivery_day_bounds(pd.Timestamp(last).date())[1]
    return pd.date_range(start, end, freq="h", inclusive="left", name="ts_utc")


def random_hourly(first: str, last: str, seed: int) -> pd.DataFrame:
    """An hourly table with random prices and forecasts, for tests that run whole models."""
    hours = hours_between(first, last)
    rng = np.random.default_rng(seed)
    count = len(hours)
    hourly = pd.DataFrame({
        "ts_utc": hours, "price": rng.normal(100, 20, count), "load_fcst": rng.uniform(13000, 25000, count),
        "solar_fcst": rng.uniform(0, 8000, count), "wind_onshore_fcst": rng.uniform(0, 9000, count),
        "wind_offshore_fcst": 0.0,
    })
    hourly["wind_fcst"] = hourly["wind_onshore_fcst"]
    hourly["residual_load_fcst"] = hourly["load_fcst"] - hourly["wind_fcst"] - hourly["solar_fcst"]
    return hourly
