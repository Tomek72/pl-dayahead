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

SERIES_COLUMNS = {  # downloaded series name -> column name in the hourly table
    "price_da": "price",
    "load_da_fcst": "load_fcst",
    "solar_da_fcst": "solar_fcst",
    "wind_onshore_da_fcst": "wind_onshore_fcst",
    "wind_offshore_da_fcst": "wind_offshore_fcst",
}
HOURLY_COLUMNS = ["ts_utc", "price", "load_fcst", "solar_fcst", "wind_onshore_fcst", "wind_offshore_fcst", "wind_fcst",
                  "residual_load_fcst"]


def build_hourly(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn the latest downloaded values into one row per hour. Returns the table and a report of missing and dropped values."""
    rows = raw[raw["series"].isin(SERIES_COLUMNS)].copy()
    report = {"ignored_series": sorted(set(raw["series"]) - set(SERIES_COLUMNS))}

    rows["ts_utc"] = ensure_utc(rows["ts_utc"])
    rows["hour"] = rows["ts_utc"].dt.floor("h")
    per_hour = rows.groupby(["series", "hour"]).agg(
        value=("value", "mean"),
        minutes=("resolution_min", "sum"),
        lengths=("resolution_min", "nunique"),
    )

    # Keep an hour only if its values are all the same length and cover all 60 minutes.
    complete = (per_hour["minutes"] == 60) & (per_hour["lengths"] == 1)
    report["dropped_hours"] = per_hour.loc[~complete].groupby(level="series").size().to_dict()
    values = per_hour.loc[complete, "value"].unstack("series").rename(columns=SERIES_COLUMNS)
    for column in SERIES_COLUMNS.values():
        if column not in values:
            values[column] = float("nan")

    # One row per hour from the first to the last, so gaps are visible as empty rows.
    hourly = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="h", name="ts_utc"))

    # Offshore wind is zero before its first forecast: the farms did not exist yet.
    first_offshore = hourly["wind_offshore_fcst"].first_valid_index()
    before = hourly.index < first_offshore if first_offshore is not None else slice(None)
    hourly.loc[before, "wind_offshore_fcst"] = 0.0

    hourly["wind_fcst"] = hourly["wind_onshore_fcst"] + hourly["wind_offshore_fcst"]
    hourly["residual_load_fcst"] = hourly["load_fcst"] - hourly["wind_fcst"] - hourly["solar_fcst"]

    hourly = hourly.reset_index()[HOURLY_COLUMNS]
    report.update(
        n_hours=len(hourly),
        first=hourly["ts_utc"].iloc[0],
        last=hourly["ts_utc"].iloc[-1],
        missing={column: int(hourly[column].isna().sum()) for column in HOURLY_COLUMNS[1:]},
        negative={column: int((hourly[column] < 0).sum()) for column in ["load_fcst", "solar_fcst", "wind_fcst"]},
        first_offshore=first_offshore,
    )
    return hourly, report


def build(con) -> dict:
    """Rebuild the hourly table in the database from the latest downloaded values. Returns the report."""
    raw = con.sql("SELECT series, ts_utc, resolution_min, value FROM entsoe_latest").df()
    hourly, report = build_hourly(raw)
    con.register("new_hourly", hourly)
    try:
        con.execute("CREATE OR REPLACE TABLE hourly AS SELECT * FROM new_hourly")
    finally:
        con.unregister("new_hourly")
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
    for column, count in report["missing"].items():
        print(f"  {column:20s} {count:6d}")
    print("hours dropped because their values do not cover 60 minutes in equal parts:", report["dropped_hours"] or "none")
    print("negative values (should be zero):", report["negative"])
    if report["ignored_series"]:
        print("downloaded series not used:", report["ignored_series"])


if __name__ == "__main__":
    main()
