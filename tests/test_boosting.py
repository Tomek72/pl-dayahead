from datetime import date

import numpy as np
import pandas as pd
from helpers import random_hourly

from pl_dayahead.backtest import walk_forward
from pl_dayahead.features import PRICE_INPUTS, build_features
from pl_dayahead.models.boosting import PARAMS, GradientBoosting, GradientBoostingDefaults, GradientBoostingPapers
from pl_dayahead.models.linear import LinearByHour


def inputs(n_days: int, seed: int) -> pd.DataFrame:
    """Random model inputs: one row per hour, every weekday, no holidays."""
    rng = np.random.default_rng(seed)
    count = 24 * n_days
    return pd.DataFrame({
        "local_hour": np.tile(np.arange(24), n_days),
        "weekday": np.repeat(np.arange(n_days) % 7, 24),
        "is_holiday": False,
        **{column: rng.normal(100, 30, count) for column in PRICE_INPUTS},
        "load_fcst": rng.uniform(13000, 25000, count),
        "res_fcst": rng.uniform(0, 15000, count),
        "res_fcst_lag_1d": rng.uniform(0, 15000, count),
    })


def merit_order_price(X: pd.DataFrame) -> pd.Series:
    """A kinked supply curve in load net of renewables: cheap and flat below 12 GW, steep above it."""
    residual_gw = (X["load_fcst"] - X["res_fcst_lag_1d"]) / 1000  # the main models' renewables input
    return pd.Series(np.where(residual_gw > 12, 60 + 25 * (residual_gw - 12), 30 + 2.5 * residual_gw), index=X.index)


def mae(prediction, actual) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(actual))))


def test_same_inputs_as_the_linear_model_plus_the_hour():
    assert set(GradientBoosting.features) == set(LinearByHour.features)  # local_hour is already listed there
    assert "local_hour" in GradientBoosting.features
    assert "res_fcst_lag_1d" in GradientBoosting.features and "res_fcst" not in GradientBoosting.features
    assert "res_fcst" in GradientBoostingPapers.features and "res_fcst_lag_1d" not in GradientBoostingPapers.features


def test_learns_a_kinked_merit_order_better_than_the_linear_model():
    train, test = inputs(150, seed=0), inputs(30, seed=1)
    target = merit_order_price(train)
    gbm = GradientBoosting().fit(train[GradientBoosting.features], target)
    linear = LinearByHour().fit(train[LinearByHour.features], target)
    gbm_error = mae(gbm.predict(test[GradientBoosting.features]), merit_order_price(test))
    linear_error = mae(linear.predict(test[LinearByHour.features]), merit_order_price(test))
    assert gbm_error < 0.5 * linear_error


def test_same_forecast_every_run():
    X = inputs(60, seed=2)
    target = merit_order_price(X)
    first = GradientBoosting().fit(X[GradientBoosting.features], target).predict(X[GradientBoosting.features])
    second = GradientBoosting().fit(X[GradientBoosting.features], target).predict(X[GradientBoosting.features])
    assert np.array_equal(first, second)


def test_the_defaults_check_changes_the_settings_and_nothing_else():
    main = GradientBoosting().model.get_params()
    defaults = GradientBoostingDefaults().model.get_params()
    changed = {key for key in main if main[key] != defaults[key]}
    assert changed == {"learning_rate", "max_iter", "min_samples_leaf", "l2_regularization"}
    assert defaults["loss"] == PARAMS["loss"] == "absolute_error"
    assert GradientBoostingDefaults.features == GradientBoosting.features


def test_both_versions_run_in_the_walk_forward_backtest():
    features, _ = build_features(random_hourly("2023-01-01", "2024-01-03", seed=3))
    for model in [GradientBoosting, GradientBoostingPapers, GradientBoostingDefaults]:  # three daily fits each
        predictions = walk_forward(features, model, first_test_day=date(2024, 1, 1))
        assert predictions["model"].eq(model.name).all()
        assert len(predictions) == 3 * 24
        assert predictions["prediction"].notna().all()
