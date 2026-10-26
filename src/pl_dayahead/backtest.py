"""Backtest: replay history one delivery day at a time.

Run from the project root, after staging:

    uv run python -m pl_dayahead.backtest naive_weekly naive_similar_day linear linear_res_d1 gbm gbm_res_d1

For each delivery day from 1 January 2024, a fresh model is trained on the 365 days ending the day
before, then forecasts that day's hours. The last training day's prices were published the afternoon
before it, so every training price was known at the 11:00 forecast time. Retraining every day follows
Lago et al. (2021) and Ziel and Weron (2018).

Forecasts are saved to the predictions table with the first and last training day, replacing the
model's earlier forecasts.
"""

from __future__ import annotations

import argparse
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
    days = features["delivery_day"].to_numpy()
    test_days = pd.DatetimeIndex(np.unique(days))
    test_days = test_days[test_days >= pd.Timestamp(first_test_day)]
    out = []
    for day in test_days:
        train_from = day - pd.Timedelta(days=train_days)
        train_to = day - pd.Timedelta(days=1)
        # The table is sorted by time, so each day's rows form one block.
        lo, start, end = np.searchsorted(days, [train_from.to_datetime64(), day.to_datetime64(),
                                                (day + pd.Timedelta(days=1)).to_datetime64()])
        train, test = features.iloc[lo:start], features.iloc[start:end]

        model = model_factory()
        cols = list(model.features)
        train = train.dropna(subset=cols + [TARGET])
        prediction = pd.Series(np.nan, index=test.index)
        usable = test[cols].notna().all(axis=1)
        if not train.empty and usable.any():
            model.fit(train[cols], train[TARGET])
            prediction[usable] = model.predict(test.loc[usable, cols])

        out.append(pd.DataFrame({
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
    return pd.concat(out, ignore_index=True)


def save(con, predictions: pd.DataFrame) -> None:
    """Write a model's forecasts to the predictions table, replacing its earlier forecasts."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            model VARCHAR, ts_utc TIMESTAMPTZ, delivery_day DATE, issue_time_utc TIMESTAMPTZ,
            trained_from DATE, trained_to DATE, prediction DOUBLE, actual DOUBLE, run_at TIMESTAMPTZ
        )
    """)
    name = predictions["model"].iloc[0]
    con.execute("DELETE FROM predictions WHERE model = ?", [name])
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
    parser.add_argument("models", nargs="+", choices=sorted(MODELS))
    args = parser.parse_args(argv)

    con = store.connect()
    hourly = con.sql("SELECT * FROM hourly ORDER BY ts_utc").df()
    features, convention = build_features(hourly)
    print(f"features: {len(features):,} hours; availability check passed")
    if convention:
        print(f"  inputs relying on the papers' convention (see availability.py): {', '.join(convention)}")
    for name in args.models:
        print(f"{name}: training every day from {FIRST_TEST_DAY}")
        preds = walk_forward(features, MODELS[name], verbose=True)
        save(con, preds)
        scored = preds.dropna(subset=["prediction", "actual"])
        mae = (scored["prediction"] - scored["actual"]).abs().mean()
        print(f"{name}: {preds['delivery_day'].nunique():,} test days, {len(scored):,} of {len(preds):,} "
              f"hours forecast, MAE {mae:.2f} EUR/MWh")
    con.close()


if __name__ == "__main__":
    main()
