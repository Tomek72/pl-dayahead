from datetime import date

import numpy as np
import pandas as pd

from pl_dayahead.backtest import walk_forward
from pl_dayahead.features import PRICE_INPUTS, build_features
from pl_dayahead.models.boosting import GradientBoosting, GradientBoostingResD1
from pl_dayahead.models.linear import LinearByHour
from pl_dayahead.timeutils import delivery_index


def inputs(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 24 * n_days
    return pd.DataFrame({
        "local_hour": np.tile(np.arange(24), n_days),
        "weekday": np.repeat(np.arange(n_days) % 7, 24),
        "is_holiday": False,
        **{col: rng.normal(100, 30, n) for col in PRICE_INPUTS},
        "load_fcst": rng.uniform(13000, 25000, n),
        "res_fcst": rng.uniform(0, 15000, n),
        "res_fcst_lag_1d": rng.uniform(0, 15000, n),
    })


def merit_order_price(X: pd.DataFrame) -> pd.Series:
    """A kinked supply curve in load net of renewables: cheap and flat below 12 GW, steep above it."""
    gw = (X["load_fcst"] - X["res_fcst"]) / 1000
    return pd.Series(np.where(gw > 12, 60 + 25 * (gw - 12), 30 + 2.5 * gw), index=X.index)


def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def test_same_inputs_as_the_linear_model_plus_the_hour():
    assert set(GradientBoosting.features) == set(LinearByHour.features)  # local_hour is already listed there
    assert "local_hour" in GradientBoosting.features
    assert "res_fcst_lag_1d" in GradientBoostingResD1.features and "res_fcst" not in GradientBoostingResD1.features


def test_learns_a_kinked_merit_order_better_than_the_linear_model():
    train, test = inputs(150, seed=0), inputs(30, seed=1)
    y = merit_order_price(train)
    gbm = GradientBoosting().fit(train[GradientBoosting.features], y)
    linear = LinearByHour().fit(train[LinearByHour.features], y)
    gbm_error = mae(gbm.predict(test[GradientBoosting.features]), merit_order_price(test))
    linear_error = mae(linear.predict(test[LinearByHour.features]), merit_order_price(test))
    assert gbm_error < 0.5 * linear_error


def test_same_forecast_every_run():
    X = inputs(60, seed=2)
    y = merit_order_price(X)
    first = GradientBoosting().fit(X[GradientBoosting.features], y).predict(X[GradientBoosting.features])
    second = GradientBoosting().fit(X[GradientBoosting.features], y).predict(X[GradientBoosting.features])
    assert np.array_equal(first, second)


def test_both_versions_run_in_the_walk_forward_backtest():
    days = pd.date_range("2023-01-01", "2024-01-03", freq="D")
    idx = delivery_index(days[0].date(), "h")
    for d in days[1:]:
        idx = idx.append(delivery_index(d.date(), "h"))
    rng = np.random.default_rng(3)
    n = len(idx)
    hourly = pd.DataFrame({
        "ts_utc": idx, "price": rng.normal(100, 20, n), "load_fcst": rng.uniform(13000, 25000, n),
        "solar_fcst": rng.uniform(0, 8000, n), "wind_onshore_fcst": rng.uniform(0, 9000, n),
        "wind_offshore_fcst": 0.0,
    })
    hourly["wind_fcst"] = hourly["wind_onshore_fcst"]
    hourly["residual_load_fcst"] = hourly["load_fcst"] - hourly["wind_fcst"] - hourly["solar_fcst"]
    feats, _ = build_features(hourly)
    for model in [GradientBoosting, GradientBoostingResD1]:  # three daily fits each
        preds = walk_forward(feats, model, first_test_day=date(2024, 1, 1))
        assert preds["model"].eq(model.name).all()
        assert len(preds) == 3 * 24
        assert preds["prediction"].notna().all()
