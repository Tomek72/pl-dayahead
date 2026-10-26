"""Gradient-boosted trees: one nonlinear model for all hours, with the linear model's inputs plus the hour.

Trees can learn that the price responds unevenly to demand: little while cheap plants cover it, and
sharply once expensive ones are needed. This model tests whether that helps when the information is
the same as the linear model's.

The settings were fixed before any backtest and never tuned. The model is trained on absolute error,
the measure it is scored on. Trees cannot forecast a price outside the range of their training data;
the linear model can. Lago et al. (2021) use a neural network as their nonlinear model; trees were
chosen because one fit takes about a second, which makes daily retraining practical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pl_dayahead.features import PRICE_INPUTS

PARAMS = dict(
    loss="absolute_error",
    learning_rate=0.05,
    max_iter=400,  # number of trees
    max_leaf_nodes=31,  # size of each tree
    min_samples_leaf=40,  # every leaf averages at least 40 training hours
    l2_regularization=1.0,
    early_stopping=False,  # always the full 400 trees, so every run is identical
    random_state=0,
)


def _features(renewables: str) -> list[str]:
    """Return the model's input columns, using the given renewables forecast column."""
    return ["local_hour", "weekday", "is_holiday", *PRICE_INPUTS, "load_fcst", renewables]


class GradientBoosting:
    """Main version: uses the renewables forecast for the delivery day, as the papers do."""

    name = "gbm"
    features = _features("res_fcst")

    def __init__(self) -> None:
        """Create an untrained model with the fixed settings in PARAMS."""
        self.model = HistGradientBoostingRegressor(**PARAMS)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoosting":
        """Train the trees on the training rows."""
        self.model.fit(X[self.features].astype(float), y.astype(float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return one forecast per row."""
        return self.model.predict(X[self.features].astype(float))


class GradientBoostingResD1(GradientBoosting):
    """Comparison version: uses the renewables forecast for the day before, so every input was published by 11:00."""

    name = "gbm_res_d1"
    features = _features("res_fcst_lag_1d")
