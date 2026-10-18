"""Walk-forward backtest: retrain every day, forecast the next delivery day.

Run from the project root, after staging:

    uv run python -m pl_dayahead.backtest naive_weekly naive_similar_day linear linear_res_d1 gbm gbm_res_d1

For each delivery day d from 1 January 2024:

1. Train a fresh model on the 365 delivery days ending d-1. Day d-1's prices were public from
   about 13:00 on d-2, so every training target was known at the 11:00 issue time on d-1.
2. Forecast the 24 hours of d (23 or 25 on clock-change days). An hour where an input the
   model needs is missing gets no forecast (NaN) rather than a guess. A training row with a
   missing input or target is dropped.

Daily retraining on a rolling window is what the papers do (Lago et al. 2021 recalibrate every
day; Ziel and Weron 2018 roll their window forward one day at a time). It lets the model follow
changes in the price level, which follows fuel and carbon costs it is not given, and every
forecast is made as if in real time.

Before a model runs, availability.check_model refuses it if it promises inputs public at 11:00
(the comparison models and the naive benchmarks) but uses one that relies on the papers'
convention.

The predictions are written to the `predictions` table in DuckDB, replacing earlier rows of the
same model, so evaluate.py and the notebooks can read them without retraining. Each row records
the first and last training day, so every forecast can be traced to the data its model saw.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from pl_dayahead import store
from pl_dayahead.availability import check_model
from pl_dayahead.features import TARGET, build_features
from pl_dayahead.models import MODELS

FIRST_TEST_DAY = date(2024, 1, 1)
TRAIN_DAYS = 365


def walk_forward(features: pd.DataFrame, model_factory, first_test_day: date = FIRST_TEST_DAY,
                 train_days: int = TRAIN_DAYS, verbose: bool = False) -> pd.DataFrame:
    """Run the backtest and return one row per forecast hour."""
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
    """Replace this model's rows in the predictions table."""
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
    parser = argparse.ArgumentParser(description="Walk-forward backtest of one or more models.")
    parser.add_argument("models", nargs="+", choices=sorted(MODELS))
    args = parser.parse_args(argv)

    con = store.connect()
    hourly = con.sql("SELECT * FROM hourly ORDER BY ts_utc").df()
    features, convention = build_features(hourly)
    print(f"features: {len(features):,} hours; availability check passed")
    if convention:
        print(f"  inputs relying on the papers' convention (see availability.py): {', '.join(convention)}")
    for name in args.models:  # refuse a mismatched model before any long run starts
        model = MODELS[name]
        check_model(name, model.features, convention, model.public_at_issue_time)

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
