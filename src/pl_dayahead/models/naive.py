"""Naive benchmarks: simple rules with nothing to learn, which every real model has to beat.

The weekly naive is the benchmark for relative MAE, as Lago et al. (2021) recommend. The similar-day
naive is reported alongside it; it is the default naive forecast in epftoolbox, the code library that
accompanies Lago et al.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK_LAG_DAYS = {0, 5, 6}  # Monday, Saturday, Sunday (weekday numbers, Monday = 0)


class NaiveWeekly:
    """Forecast each hour as the price at the same hour one week earlier."""

    name = "naive_weekly"
    features = ["price_lag_7d"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveWeekly":
        """Nothing to learn: return the model unchanged."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the price at the same hour one week earlier for each row."""
        return X["price_lag_7d"].to_numpy()


class NaiveSimilarDay:
    """Forecast each hour as the price at the same hour the day before, or one week earlier for Saturday, Sunday and Monday.

    For those three days the day before is a different kind of day, so last week's price is a better guide.
    """

    name = "naive_similar_day"
    features = ["weekday", "price_lag_1d", "price_lag_7d"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveSimilarDay":
        """Nothing to learn: return the model unchanged."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return yesterday's price at the same hour from Tuesday to Friday, and last week's on Saturday, Sunday and Monday."""
        use_week = X["weekday"].isin(WEEK_LAG_DAYS)
        return np.where(use_week, X["price_lag_7d"], X["price_lag_1d"])
