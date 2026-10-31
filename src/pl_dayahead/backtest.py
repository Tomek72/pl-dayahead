"""Backtest: replay history one delivery day at a time.

Run from the project root, after staging, to backtest every registered model (name some models
after the command to run only those):

    uv run python -m pl_dayahead.backtest

For each delivery day from 1 January 2024, a fresh model is trained on the 365 days ending the day
before, then forecasts that day's hours. The last training day's prices were published the afternoon
before it, so every training price was known at the 11:00 forecast time. Retraining every day follows
Lago et al. (2021) and Ziel and Weron (2018).

Forecasts are saved to the predictions table with the first and last training day, replacing the
model's earlier forecasts.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

from pl_dayahead import store
from pl_dayahead.features import TARGET, build_features
from pl_dayahead.models import MODELS

FIRST_TEST_DAY = date(2024, 1, 1)
TRAIN_DAYS = 365


def walk_forward(features: pd.DataFrame, model_factory, first_test_day: date = FIRST_TEST_DAY,
                 train_days: int = TRAIN_DAYS, verbose: bool = False) -> pd.DataFrame:
    """Run the backtest for one model and return its forecasts.

    For each delivery day from `first_test_day`, train a fresh model on the `train_days` days before
    it, dropping rows with a missing input or price, then forecast that day's hours. Hours with a
    missing input get no forecast. Returns one row per hour with the model name, times, first and
    last training day, forecast and actual price.
    """
    features = features.sort_values("ts_utc").reset_index(drop=True)
    delivery_days = features["delivery_day"].to_numpy()
    test_days = pd.DatetimeIndex(np.unique(delivery_days))
    test_days = test_days[test_days >= pd.Timestamp(first_test_day)]
    forecasts_by_day = []
    for day in test_days:
        train_from = day - pd.Timedelta(days=train_days)
        train_to = day - pd.Timedelta(days=1)
        # The table is sorted by time, so each day's rows form one block.
        train_start, test_start, test_end = np.searchsorted(
            delivery_days, [train_from.to_datetime64(), day.to_datetime64(), (day + pd.Timedelta(days=1)).to_datetime64()])
        train, test = features.iloc[train_start:test_start], features.iloc[test_start:test_end]

        model = model_factory()
        inputs = list(model.features)
        train = train.dropna(subset=inputs + [TARGET])
        prediction = pd.Series(np.nan, index=test.index)
        usable = test[inputs].notna().all(axis=1)
        if not train.empty and usable.any():
            model.fit(train[inputs], train[TARGET])
            prediction[usable] = model.predict(test.loc[usable, inputs])

        forecasts_by_day.append(pd.DataFrame({
            "model": model.name,
            "ts_utc": test["ts_utc"],
            "delivery_day": test["delivery_day"],
            "issue_time_utc": test["issue_time_utc"],
            "trained_from": train_from,
            "trained_to": train_to,
            "prediction": prediction,
            "actual": test[TARGET],
        }))
        if verbose and (day.day == 1 or day == test_days[-1]):
            print(f"  {model.name}: {day:%Y-%m-%d}", flush=True)
    return pd.concat(forecasts_by_day, ignore_index=True)


def save(con, predictions: pd.DataFrame) -> None:
    """Write a model's forecasts to the predictions table, replacing its earlier forecasts."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            model VARCHAR, ts_utc TIMESTAMPTZ, delivery_day DATE, issue_time_utc TIMESTAMPTZ,
            trained_from DATE, trained_to DATE, prediction DOUBLE, actual DOUBLE, run_at TIMESTAMPTZ
        )
    """)
    con.execute("DELETE FROM predictions WHERE model = ?", [predictions["model"].iloc[0]])
    con.register("new_predictions", predictions.assign(run_at=pd.Timestamp.now(tz="UTC")))
    try:
        con.execute("""
            INSERT INTO predictions
            SELECT model, ts_utc, CAST(delivery_day AS DATE), issue_time_utc,
                   CAST(trained_from AS DATE), CAST(trained_to AS DATE), prediction, actual, run_at
            FROM new_predictions
        """)
    finally:
        con.unregister("new_predictions")


def main(argv: list[str] | None = None) -> None:
    """Read the model names from the command line, then backtest and save each one."""
    parser = argparse.ArgumentParser(description="Walk-forward backtest of one or more models.")
    parser.add_argument("models", nargs="*", help=f"models to run, all if none are named: {', '.join(MODELS)}")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.models) - set(MODELS))
    if unknown:
        parser.error(f"unknown models: {', '.join(unknown)}")

    con = store.connect()
    hourly = con.sql("SELECT * FROM hourly ORDER BY ts_utc").df()
    auctions = con.sql("SELECT auction_date, price, volume FROM eua_auctions").df()
    features, convention = build_features(hourly, auctions)
    print(f"features: {len(features):,} hours; availability check passed")
    if convention:
        print(f"  inputs relying on the papers' convention (see availability.py): {', '.join(convention)}")
    names = args.models or list(MODELS)
    # A carbon variant must forecast the same hours as its base model, or every model's scores move
    # to fewer common hours, so a gap in the auctions stops the run instead.
    if any("eua_lag_2d" in MODELS[name].features for name in names) and features["eua_lag_2d"].isna().any():
        missing = features.loc[features["eua_lag_2d"].isna(), "delivery_day"]
        con.close()
        sys.exit(f"no EUA price for {missing.nunique()} delivery days from {missing.min():%Y-%m-%d}: "
                 "run python -m pl_dayahead.ingest.eex first")
    for name in names:
        print(f"{name}: training every day from {FIRST_TEST_DAY}")
        predictions = walk_forward(features, MODELS[name], verbose=True)
        save(con, predictions)
        scored = predictions.dropna(subset=["prediction", "actual"])
        mae = (scored["prediction"] - scored["actual"]).abs().mean()
        print(f"{name}: {predictions['delivery_day'].nunique():,} test days, {len(scored):,} of {len(predictions):,} "
              f"hours forecast, MAE {mae:.2f} EUR/MWh")
    con.close()


if __name__ == "__main__":
    main()
