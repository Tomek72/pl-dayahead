from datetime import date

import numpy as np
import pandas as pd

from pl_dayahead.backtest import walk_forward
from pl_dayahead.features import PRICE_INPUTS, build_features
from pl_dayahead.models.linear import LinearByHour, LinearResD1, latest_coefficients
from pl_dayahead.timeutils import delivery_index


def inputs(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Random model inputs: one row per hour, every weekday and some holidays.

    As in real data, the hour-23 row's last price of d-1 equals its same-hour lag.
    """
    rng = np.random.default_rng(seed)
    n = 24 * n_days
    hour = np.tile(np.arange(24), n_days)
    X = pd.DataFrame({
        "local_hour": hour,
        "weekday": np.repeat(np.arange(n_days) % 7, 24),
        "is_holiday": np.repeat(rng.random(n_days) < 0.1, 24),
        **{col: rng.normal(100, 30, n) for col in PRICE_INPUTS},
        "load_fcst": rng.uniform(13000, 25000, n),
        "res_fcst": rng.uniform(0, 20000, n),
        "res_fcst_lag_1d": rng.uniform(0, 20000, n),
    })
    last_hour = X["local_hour"] == 23
    X.loc[last_hour, "price_last_1d"] = X.loc[last_hour, "price_lag_1d"]
    return X


def true_price(X: pd.DataFrame, renewables: str = "res_fcst") -> pd.Series:
    """Prices built from known slopes; renewables matter most at midday, load in the evening."""
    h = X["local_hour"]
    load_slope = np.where(h.between(17, 20), 6.0, 3.0)
    res_slope = np.where(h.between(10, 15), -8.0, -2.0)
    return (20 + 0.5 * X["price_lag_1d"] + 0.1 * X["price_lag_2d"] + 0.2 * X["price_lag_7d"]
            + 0.1 * X["price_last_1d"] + 0.05 * X["price_min_1d"] + 0.03 * X["price_max_1d"]
            + load_slope * X["load_fcst"] / 1000 + res_slope * X[renewables] / 1000
            - 10.0 * (X["weekday"] == 5) - 15.0 * (X["weekday"] == 6) - 25.0 * X["is_holiday"])


def test_sixteen_coefficients_per_hour():
    X = inputs()
    coef = LinearByHour().fit(X, true_price(X)).coefficients()
    assert coef.shape == (24, 16)
    assert "wednesday" not in coef.columns  # the baseline day


def test_recovers_hour_specific_slopes():
    X = inputs()
    coef = LinearByHour().fit(X, true_price(X)).coefficients().drop(index=23)
    assert np.isclose(coef.loc[19, "load_gw"], 6.0) and np.isclose(coef.loc[4, "load_gw"], 3.0)
    assert np.isclose(coef.loc[12, "res_gw"], -8.0) and np.isclose(coef.loc[2, "res_gw"], -2.0)
    assert np.allclose(coef["price_lag_1d"], 0.5) and np.allclose(coef["price_last_1d"], 0.1)
    assert np.allclose(coef["price_min_1d"], 0.05) and np.allclose(coef["price_max_1d"], 0.03)
    assert np.allclose(coef["saturday"], -10.0) and np.allclose(coef["sunday"], -15.0)
    assert np.allclose(coef["monday"], 0.0, atol=1e-6) and np.allclose(coef["holiday"], -25.0)


def test_predictions_match_on_new_days():
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    model = LinearByHour().fit(train, true_price(train))
    assert np.allclose(model.predict(test), true_price(test))


def test_hour_23_with_identical_last_price_and_lag():
    """The duplicate column gets the minimum-norm split of one effect, and forecasts stay exact."""
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    model = LinearByHour().fit(train, true_price(train))
    hour23 = model.coefficients().loc[23]
    assert np.isclose(hour23["price_lag_1d"] + hour23["price_last_1d"], 0.6)  # 0.5 + 0.1 combined
    assert np.isclose(hour23["price_lag_1d"], hour23["price_last_1d"])
    at23 = (test["local_hour"] == 23).to_numpy()
    assert np.allclose(model.predict(test)[at23], true_price(test)[at23])


class MondayBaseline(LinearByHour):
    """The same model with Monday instead of Wednesday as the dropped weekday."""

    def design(self, X):
        d = super().design(X).drop(columns="monday")
        d["wednesday"] = (X["weekday"] == 2).astype(float)
        return d


def test_the_choice_of_baseline_weekday_does_not_change_forecasts():
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    y = true_price(train) + np.random.default_rng(9).normal(0, 10, len(train))  # noisy, so the fit is not exact
    a = LinearByHour().fit(train, y).predict(test)
    b = MondayBaseline().fit(train, y).predict(test)
    assert np.allclose(a, b)


def test_comparison_version_uses_the_previous_day_renewables():
    X = inputs()
    y = true_price(X, renewables="res_fcst_lag_1d")
    model = LinearResD1().fit(X, y)
    assert np.allclose(model.design(X)["res_gw"], X["res_fcst_lag_1d"] / 1000)
    assert np.isclose(model.coefficients().loc[12, "res_gw"], -8.0)
    assert "res_fcst" not in LinearResD1.features and "res_fcst_lag_1d" in LinearResD1.features


def test_an_hour_never_trained_gets_no_forecast():
    X = inputs()
    model = LinearByHour().fit(X[X["local_hour"] != 2], true_price(X)[X["local_hour"] != 2])
    pred = model.predict(X)
    assert np.isnan(pred[X["local_hour"] == 2]).all()
    assert np.isfinite(pred[X["local_hour"] != 2]).all()


def features_table() -> pd.DataFrame:
    days = pd.date_range("2023-01-01", "2024-02-29", freq="D")
    idx = delivery_index(days[0].date(), "h")
    for d in days[1:]:
        idx = idx.append(delivery_index(d.date(), "h"))
    rng = np.random.default_rng(2)
    n = len(idx)
    hourly = pd.DataFrame({
        "ts_utc": idx, "price": rng.normal(100, 20, n), "load_fcst": rng.uniform(13000, 25000, n),
        "solar_fcst": rng.uniform(0, 8000, n), "wind_onshore_fcst": rng.uniform(0, 9000, n),
        "wind_offshore_fcst": 0.0,
    })
    hourly["wind_fcst"] = hourly["wind_onshore_fcst"]
    hourly["residual_load_fcst"] = hourly["load_fcst"] - hourly["wind_fcst"] - hourly["solar_fcst"]
    feats, _ = build_features(hourly)
    return feats


def test_both_versions_run_in_the_walk_forward_backtest():
    feats = features_table()
    for model in [LinearByHour, LinearResD1]:
        preds = walk_forward(feats, model, first_test_day=date(2024, 2, 1))
        assert preds["model"].eq(model.name).all()
        assert preds["prediction"].notna().all()
        assert preds["delivery_day"].nunique() == 29


def test_latest_coefficients_use_the_last_year():
    table, first, last = latest_coefficients(features_table())
    assert last == pd.Timestamp("2024-02-29")
    assert first == pd.Timestamp("2023-03-02")  # 365 days including the last one
    assert list(table.index) == list(range(24))
