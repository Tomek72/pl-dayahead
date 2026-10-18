"""Gradient-boosted trees: one nonlinear model for all hours of the day, on the linear model's inputs.

The linear model assumes each input moves the price by a fixed amount per unit. The merit
order says otherwise: when load net of renewables is low, the next GW is covered by cheap coal
and the price barely moves; when it is high, the next GW needs a gas peaker and the price
jumps. The supply curve is flat, then steep. Trees can learn that shape from the data without
being told. The question this model answers: given exactly the same information as the linear
model, does a nonlinear model forecast better?

How it works. A regression tree splits the training hours by questions such as "load above
22 GW?" and "hour after 17?", and predicts a single price for each group it ends with.
Boosting builds a few hundred small trees one after another, each fitted to the errors the
earlier ones still make, and adds up their predictions. One model covers all 24 hours because
the trees can split on the hour themselves, which also lets hours share what they have in
common. Twenty-four separate tree models would each have only 365 training rows.

Inputs: the linear model's inputs (price lags, last price, minimum and maximum of d-1, load
and renewables forecasts, weekday, holiday) plus the local hour. The weekday enters as one
number from 0 to 6, which trees can split on directly.

Settings, each fixed before looking at any backtest result:

* loss="absolute_error": the model is trained to minimise the same thing it is scored on
  (MAE), so a few spike hours do not drag the fit around the way they do with least squares.
* learning rate, number of trees and leaf sizes are common starting values, not tuned. Tuning
  them against the same 2024 to 2026 results we report would make the comparison unfair to the
  linear model, which has nothing to tune.

What trees cannot do: predict a price outside the range seen in training. A tree's output is an
average of training prices, so if prices fall to a level the last 365 days never saw, the
forecast stays inside the old range. The linear model has no such limit.

Lago et al. (2021) use a neural network as their nonlinear benchmark. Trees were chosen
because one fit takes about a second, which makes daily retraining practical, and they need no
architecture search. This uses scikit-learn's HistGradientBoostingRegressor, a histogram-based
implementation of the same method as LightGBM.
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
    return ["local_hour", "weekday", "is_holiday", *PRICE_INPUTS, "load_fcst", renewables]


class GradientBoosting:
    name = "gbm"
    features = _features("res_fcst")  # renewables for day d: the papers' convention
    public_at_issue_time = False

    def __init__(self) -> None:
        self.model = HistGradientBoostingRegressor(**PARAMS)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoosting":
        self.model.fit(X[self.features].astype(float), y.astype(float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features].astype(float))


class GradientBoostingResD1(GradientBoosting):
    """Comparison version: the renewables forecast for d-1, so every input was public at 11:00."""

    name = "gbm_res_d1"
    features = _features("res_fcst_lag_1d")
    public_at_issue_time = True
