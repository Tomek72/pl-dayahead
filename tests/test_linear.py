from datetime import date

import numpy as np
import pandas as pd
from helpers import random_hourly

from pl_dayahead.backtest import walk_forward
from pl_dayahead.features import PRICE_INPUTS, build_features
from pl_dayahead.models.linear import LinearAsinh, LinearByHour, LinearCarbon, LinearPapers, latest_coefficients


def inputs(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    """Random model inputs: one row per hour, every weekday and some holidays.

    As in real data, the hour-23 row's last price of d-1 equals its same-hour lag.
    """
    rng = np.random.default_rng(seed)
    count = 24 * n_days
    hour = np.tile(np.arange(24), n_days)
    X = pd.DataFrame({
        "local_hour": hour,
        "weekday": np.repeat(np.arange(n_days) % 7, 24),
        "is_holiday": np.repeat(rng.random(n_days) < 0.1, 24),
        **{column: rng.normal(100, 30, count) for column in PRICE_INPUTS},
        "load_fcst": rng.uniform(13000, 25000, count),
        "renewables_fcst": rng.uniform(0, 20000, count),
        "renewables_fcst_lag_1d": rng.uniform(0, 20000, count),
    })
    last_hour = X["local_hour"] == 23
    X.loc[last_hour, "price_last_1d"] = X.loc[last_hour, "price_lag_1d"]
    return X


def true_price(X: pd.DataFrame, renewables: str = "renewables_fcst_lag_1d") -> pd.Series:
    """Prices built from known slopes; renewables matter most at midday, load in the evening."""
    hour = X["local_hour"]
    load_slope = np.where(hour.between(17, 20), 6.0, 3.0)
    renewables_slope = np.where(hour.between(10, 15), -8.0, -2.0)
    return (20 + 0.5 * X["price_lag_1d"] + 0.1 * X["price_lag_2d"] + 0.2 * X["price_lag_7d"]
            + 0.1 * X["price_last_1d"] + 0.05 * X["price_min_1d"] + 0.03 * X["price_max_1d"]
            + load_slope * X["load_fcst"] / 1000 + renewables_slope * X[renewables] / 1000
            - 10.0 * (X["weekday"] == 5) - 15.0 * (X["weekday"] == 6) - 25.0 * X["is_holiday"])


def test_sixteen_coefficients_per_hour():
    X = inputs()
    coefficients = LinearByHour().fit(X, true_price(X)).coefficients()
    assert coefficients.shape == (24, 16)
    assert "wednesday" not in coefficients.columns  # the baseline day


def test_carbon_version_adds_a_seventeenth_coefficient_and_recovers_its_slope():
    X = inputs()
    X["eua_lag_2d"] = np.repeat(np.random.default_rng(5).uniform(60, 90, 60), 24)  # one price per day
    coefficients = LinearCarbon().fit(X, true_price(X) + 0.4 * X["eua_lag_2d"]).coefficients()
    assert coefficients.shape == (24, 17)
    assert np.allclose(coefficients["eua"], 0.4)


def test_recovers_hour_specific_slopes():
    X = inputs()
    coefficients = LinearByHour().fit(X, true_price(X)).coefficients().drop(index=23)
    assert np.isclose(coefficients.loc[19, "load_gw"], 6.0) and np.isclose(coefficients.loc[4, "load_gw"], 3.0)
    assert np.isclose(coefficients.loc[12, "renewables_gw"], -8.0)
    assert np.isclose(coefficients.loc[2, "renewables_gw"], -2.0)
    assert np.allclose(coefficients["price_lag_1d"], 0.5) and np.allclose(coefficients["price_last_1d"], 0.1)
    assert np.allclose(coefficients["price_min_1d"], 0.05) and np.allclose(coefficients["price_max_1d"], 0.03)
    assert np.allclose(coefficients["saturday"], -10.0) and np.allclose(coefficients["sunday"], -15.0)
    assert np.allclose(coefficients["monday"], 0.0, atol=1e-6) and np.allclose(coefficients["holiday"], -25.0)


def test_predictions_match_on_new_days():
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    model = LinearByHour().fit(train, true_price(train))
    assert np.allclose(model.predict(test), true_price(test))


def test_hour_23_with_identical_last_price_and_lag():
    """The duplicate column gets the minimum-norm split of one effect, and forecasts stay exact."""
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    model = LinearByHour().fit(train, true_price(train))
    hour_23 = model.coefficients().loc[23]
    assert np.isclose(hour_23["price_lag_1d"] + hour_23["price_last_1d"], 0.6)  # 0.5 + 0.1 combined
    assert np.isclose(hour_23["price_lag_1d"], hour_23["price_last_1d"])
    at_23 = (test["local_hour"] == 23).to_numpy()
    assert np.allclose(model.predict(test)[at_23], true_price(test)[at_23])


class MondayBaseline(LinearByHour):
    """The same model with Monday instead of Wednesday as the dropped weekday."""

    def design(self, X):
        regressors = super().design(X).drop(columns="monday")
        regressors["wednesday"] = (X["weekday"] == 2).astype(float)
        return regressors


def test_the_choice_of_baseline_weekday_does_not_change_forecasts():
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    target = true_price(train) + np.random.default_rng(9).normal(0, 10, len(train))  # noisy, so the fit is not exact
    wednesday_baseline = LinearByHour().fit(train, target).predict(test)
    monday_baseline = MondayBaseline().fit(train, target).predict(test)
    assert np.allclose(wednesday_baseline, monday_baseline)


def test_main_version_uses_the_previous_day_renewables_and_the_papers_setup_day_d():
    X = inputs()
    main = LinearByHour().fit(X, true_price(X))
    assert np.allclose(main.design(X)["renewables_gw"], X["renewables_fcst_lag_1d"] / 1000)
    assert "renewables_fcst" not in LinearByHour.features and "renewables_fcst_lag_1d" in LinearByHour.features
    papers = LinearPapers().fit(X, true_price(X, renewables="renewables_fcst"))
    assert np.allclose(papers.design(X)["renewables_gw"], X["renewables_fcst"] / 1000)
    assert np.isclose(papers.coefficients().loc[12, "renewables_gw"], -8.0)
    assert "renewables_fcst" in LinearPapers.features and "renewables_fcst_lag_1d" not in LinearPapers.features


def test_asinh_version_centres_on_the_median_and_scales_by_the_mad():
    X = inputs()
    target = true_price(X)
    model = LinearAsinh().fit(X, target)
    assert model.centre == target.median()
    assert np.isclose(model.scale, 1.4826 * (target - target.median()).abs().median())


def test_asinh_version_is_pulled_less_by_spikes():
    """A few huge prices in training move the least-squares forecasts far more than the asinh ones."""
    train, test = inputs(seed=0), inputs(n_days=7, seed=1)
    target = true_price(train)
    spiky = target.copy()
    spiky.iloc[::100] += 2000.0  # one hour in a hundred about 2,000 EUR/MWh higher
    shift = {}
    for model in [LinearByHour, LinearAsinh]:
        clean = model().fit(train, target).predict(test)
        shift[model.name] = np.mean(np.abs(model().fit(train, spiky).predict(test) - clean))
    assert shift["linear_asinh"] < 0.2 * shift["linear"]


def test_an_hour_never_trained_gets_no_forecast():
    X = inputs()
    model = LinearByHour().fit(X[X["local_hour"] != 2], true_price(X)[X["local_hour"] != 2])
    forecast = model.predict(X)
    assert np.isnan(forecast[X["local_hour"] == 2]).all()
    assert np.isfinite(forecast[X["local_hour"] != 2]).all()


def features_table() -> pd.DataFrame:
    return build_features(random_hourly("2023-01-01", "2024-02-29", seed=2))[0]


def test_every_version_runs_in_the_walk_forward_backtest():
    features = features_table()
    for model in [LinearByHour, LinearPapers, LinearAsinh]:
        predictions = walk_forward(features, model, first_test_day=date(2024, 2, 1))
        assert predictions["model"].eq(model.name).all()
        assert predictions["prediction"].notna().all()
        assert predictions["delivery_day"].nunique() == 29


def test_latest_coefficients_use_the_last_year():
    table, first, last = latest_coefficients(features_table())
    assert last == pd.Timestamp("2024-02-29")
    assert first == pd.Timestamp("2023-03-02")  # 365 days including the last one
    assert list(table.index) == list(range(24))
