"""Polish public holidays, taken from the holidays package."""

from __future__ import annotations

import holidays
import pandas as pd


def is_holiday(days: pd.Series) -> pd.Series:
    """Return True for each delivery day that is a Polish public holiday."""
    dates = pd.to_datetime(days)
    poland = holidays.Poland(years=dates.dt.year.unique().tolist())
    return dates.dt.date.isin(list(poland))