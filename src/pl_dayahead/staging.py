"""Build the hourly table: one row per UTC hour with the price and the load, wind and solar forecasts.

Run from the project root after downloading:

    uv run python -m pl_dayahead.staging

Quarter-hour values are averaged into hours. An hour counts only if its values are all the same
length and together cover 60 minutes; otherwise it is left empty. Wind is onshore plus offshore, with
offshore counted as zero before its first forecast. Residual load (load minus wind and solar) is added
for the merit-order notebook. Every hour appears as a row, so gaps show up as empty rows.
"""

from __future__ import annotations

import pandas as pd

from pl_dayahead import store
from pl_dayahead.timeutils import ensure_utc

# downloaded series name -> column name in the hourly table
COLUMNS = {
    "price_da": "price",
    "load_da_fcst": "load_fcst",
    "solar_da_fcst": "solar_fcst",
    "wind_onshore_da_fcst": "wind_onshore_fcst",
    "wind_offshore_da_fcst": "wind_offshore_fcst",
}
OUTPUT = [
    "ts_utc",
    "price",
    "load_fcst",
    "solar_fcst",
    "wind_onshore_fcst",
    "wind_offshore_fcst",
    "wind_fcst",
    "residual_load_fcst",
]


def build_hourly(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the latest downloaded values into one row per hour. Returns the table and a report of missing and dropped values."""
    df = raw[raw["series"].isin(COLUMNS)].copy()
    report = {"ignored_series": sorted(set(raw["series"]) - set(COLUMNS))}

    df["ts_utc"] = ensure_utc(df["ts_utc"])
    df["hour"] = df["ts_utc"].dt.floor("h")
    per_hour = df.groupby(["series", "hour"]).agg(
        value=("value", "mean"),
        minutes=("resolution_min", "sum"),
        lengths=("resolution_min", "nunique"),
    )

    # Keep an hour only if its values are all the same length and cover all 60 minutes.
    complete = (per_hour["minutes"] == 60) & (per_hour["lengths"] == 1)
    report["dropped_hours"] = per_hour.loc[~complete].groupby(level="series").size().to_dict()
    wide = per_hour.loc[complete, "value"].unstack("series").rename(columns=COLUMNS)
    for col in COLUMNS.values():
        if col not in wide:
            wide[col] = float("nan")

    # One row per hour from the first to the last, so gaps are visible as empty rows.
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="h", name="ts_utc")
    hourly = wide.reindex(grid)

    # Offshore wind is zero before its first forecast: the farms did not exist yet.
    offshore = hourly["wind_offshore_fcst"]
    first_offshore = offshore.first_valid_index()
    before = hourly.index < first_offshore if first_offshore is not None else slice(None)
    hourly.loc[before, "wind_offshore_fcst"] = 0.0

    hourly["wind_fcst"] = hourly["wind_onshore_fcst"] + hourly["wind_offshore_fcst"]
    hourly["residual_load_fcst"] = hourly["load_fcst"] - hourly["wind_fcst"] - hourly["solar_fcst"]

    hourly = hourly.reset_index()[OUTPUT]
    value_cols = OUTPUT[1:]
    report.update(
        n_hours=len(hourly),
        first=hourly["ts_utc"].iloc[0],
        last=hourly["ts_utc"].iloc[-1],
        missing={c: int(hourly[c].isna().sum()) for c in value_cols},
        negative={c: int((hourly[c] < 0).sum()) for c in ["load_fcst", "solar_fcst", "wind_fcst"]},
        first_offshore=first_offshore,
    )
    return hourly, report


def build(con) -> dict:
    """Rebuild the hourly table in the database from the latest downloaded values. Returns the report."""
    raw = con.sql("SELECT series, ts_utc, resolution_min, value FROM entsoe_latest").df()
    hourly, report = build_hourly(raw)
    con.register("staged", hourly)
    try:
        con.execute("CREATE OR REPLACE TABLE hourly AS SELECT * FROM staged")
    finally:
        con.unregister("staged")
    return report


def main() -> None:
    """Rebuild the hourly table and print the report."""
    con = store.connect()
    report = build(con)
    con.close()

    print(f"hourly: {report['n_hours']:,} hours, {report['first']} to {report['last']} (UTC)")
    if report["first_offshore"] is not None:
        print(f"offshore wind counted as zero before {report['first_offshore']}")
    print("missing hours per column:")
    for col, n in report["missing"].items():
        print(f"  {col:20s} {n:6d}")
    print("hours dropped because their values do not cover 60 minutes in equal parts:", report["dropped_hours"] or "none")
    print("negative values (should be zero):", report["negative"])
    if report["ignored_series"]:
        print("downloaded series not used:", report["ignored_series"])


if __name__ == "__main__":
    main()
