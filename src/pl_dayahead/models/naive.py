"""Benchmarks: rules that learn nothing, which every real model must beat.

NaiveWeekly copies the same hour one week earlier. It is the benchmark for relative MAE in
evaluate.py: Lago, Marcjasz, De Schutter and Weron (2021) recommend the weekly naive forecast
for that role and call it "arguably the best choice".

NaiveSimilarDay is an extra row. For Tuesday to Friday it copies the same hour of the previous
day, because one working day looks like the next. For Saturday, Sunday and Monday the previous
day is a different kind of day (Friday, Saturday, Sunday), so it copies the same hour one week
earlier instead. It is the default naive forecast in epftoolbox, the code library that comes
with Lago et al. (2021).

Both use only prices that were public at the 11:00 issue time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK_LAG_DAYS = {0, 5, 6}  # Monday, Saturday, Sunday (weekday numbers, Monday = 0)


class NaiveWeekly:
    name = "naive_weekly"
    features = ["price_lag_7d"]
    public_at_issue_time = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveWeekly":
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["price_lag_7d"].to_numpy()


class NaiveSimilarDay:
    name = "naive_similar_day"
    features = ["weekday", "price_lag_1d", "price_lag_7d"]
    public_at_issue_time = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveSimilarDay":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        use_week = X["weekday"].isin(WEEK_LAG_DAYS)
        return np.where(use_week, X["price_lag_7d"], X["price_lag_1d"])
