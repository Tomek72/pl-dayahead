from datetime import date

import numpy as np
import pandas as pd
from helpers import hours_between

from pl_dayahead import store
from pl_dayahead.backtest import save, walk_forward
from pl_dayahead.features import build_features
from pl_dayahead.models.naive import NaiveSimilarDay, NaiveWeekly


def feature_table(first: str = "2023-01-01", last: str = "2024-03-31") -> pd.DataFrame:
    """Inputs built from random prices and constant forecasts."""
    hours = hours_between(first, last)
    hourly = pd.DataFrame({
        "ts_utc": hours, "price": np.random.default_rng(0).normal(100, 20, len(hours)), "load_fcst": 20000.0,
        "solar_fcst": 3000.0, "wind_onshore_fcst": 4000.0, "wind_offshore_fcst": 0.0, "wind_fcst": 4000.0,
        "residual_load_fcst": 13000.0,
    })
    return build_features(hourly)[0]


class Spy:
    """Records the delivery days it was trained on and the day it forecast."""
    name = "spy"
    features = ["price_lag_1d"]
    trained: list = []
    forecast: list = []

    def __init__(self, table):
        self.table = table

    def fit(self, X, y):
        Spy.trained.append(self.table.loc[X.index, "delivery_day"])
        return self

    def predict(self, X):
        Spy.forecast.append(self.table.loc[X.index, "delivery_day"].unique())
        return np.zeros(len(X))


def test_naive_predicts_the_weekly_lag():
    features = feature_table()
    predictions = walk_forward(features, NaiveWeekly, first_test_day=date(2024, 1, 1))
    merged = predictions.merge(features[["ts_utc", "price_lag_7d"]], on="ts_utc")
    assert predictions["model"].eq("naive_weekly").all()
    assert np.allclose(merged["prediction"], merged["price_lag_7d"])


def test_similar_day_naive_uses_yesterday_midweek_and_last_week_otherwise():
    features = feature_table()
    predictions = walk_forward(features, NaiveSimilarDay, first_test_day=date(2024, 1, 1))
    merged = predictions.merge(features[["ts_utc", "weekday", "price_lag_1d", "price_lag_7d"]], on="ts_utc")
    midweek = merged["weekday"].between(1, 4)  # Tuesday to Friday
    assert predictions["model"].eq("naive_similar_day").all()
    assert np.allclose(merged.loc[midweek, "prediction"], merged.loc[midweek, "price_lag_1d"])
    assert np.allclose(merged.loc[~midweek, "prediction"], merged.loc[~midweek, "price_lag_7d"])
    assert set(merged.loc[~midweek, "weekday"]) == {0, 5, 6}


def test_a_fresh_model_every_day_trained_on_the_365_days_before_it():
    features = feature_table(first="2022-12-20", last="2024-01-20")  # so no training row lacks its lag
    Spy.trained, Spy.forecast = [], []
    predictions = walk_forward(features, lambda: Spy(features), first_test_day=date(2024, 1, 1), train_days=365)
    test_days = pd.date_range("2024-01-01", "2024-01-20", freq="D")
    assert len(Spy.trained) == len(Spy.forecast) == len(test_days)  # one fit per day
    for day, trained, forecast in zip(test_days, Spy.trained, Spy.forecast):
        assert list(forecast) == [day]
        assert trained.max() == day - pd.Timedelta(days=1)  # the last training day is d-1
        assert trained.min() == day - pd.Timedelta(days=365)
    assert (predictions["trained_to"] == predictions["delivery_day"] - pd.Timedelta(days=1)).all()
    assert (predictions["trained_from"] == predictions["delivery_day"] - pd.Timedelta(days=365)).all()


def test_only_days_from_the_first_test_day_are_forecast():
    predictions = walk_forward(feature_table(), NaiveWeekly, first_test_day=date(2024, 2, 10))
    assert predictions["delivery_day"].min() == pd.Timestamp("2024-02-10")
    assert predictions["delivery_day"].max() == pd.Timestamp("2024-03-31")
    assert predictions["delivery_day"].nunique() == 51


def test_save_replaces_a_models_rows():
    con = store.connect(":memory:")
    predictions = walk_forward(feature_table(), NaiveWeekly, first_test_day=date(2024, 3, 1))
    save(con, predictions)
    save(con, predictions)  # a rerun replaces the rows instead of adding a second copy
    assert con.sql("SELECT count(*) FROM predictions").fetchone()[0] == len(predictions)
    first = con.sql("SELECT delivery_day, trained_from, trained_to FROM predictions ORDER BY ts_utc LIMIT 1")
    assert first.fetchone() == (date(2024, 3, 1), date(2023, 3, 2), date(2024, 2, 29))
    con.close()


def test_missing_feature_gives_no_forecast():
    features = feature_table()
    hole = features["delivery_day"] == pd.Timestamp("2024-01-10")
    features.loc[hole, "price_lag_7d"] = np.nan
    predictions = walk_forward(features, NaiveWeekly, first_test_day=date(2024, 1, 1))
    on_day = predictions["delivery_day"] == pd.Timestamp("2024-01-10")
    assert predictions.loc[on_day, "prediction"].isna().all()
    assert predictions.loc[~on_day, "prediction"].notna().all()
