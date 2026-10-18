"""Turn the raw ENTSO-E downloads into one clean hourly table.

Run from the project root after the ingest:

    uv run python -m pl_dayahead.staging

It reads the entsoe_latest view, builds one row per UTC hour, and writes the result to
the `hourly` table in the same DuckDB file, replacing any previous version. Every later
step (features, models, notebooks) reads `hourly` and never the raw tables.

Rules:

* Each hourly value is the time-weighted mean of the rows inside that hour: one hourly
  row is used as it is, four 15-minute rows are averaged. Forecasts are in MW, which is
  average power over the hour, so the hourly mean in MW also equals the hour's energy in MWh.
* An hour is used only if its rows cover exactly 60 minutes. A partly covered hour becomes
  missing (NaN) instead of an average of whatever is there, and is counted in the report.
* Wind is onshore plus offshore. Offshore counts as zero before its first forecast, because
  there were no offshore wind farms then; after that, a missing offshore value makes wind missing.
* Residual load is the load forecast minus the wind and solar forecasts: the demand left
  for coal, gas and other controllable plants.
* Every hour between the first and last one appears as a row, so a gap shows up as a row
  of NaN rather than as a silently missing timestamp.
"""

from __future__ import annotations

import pandas as pd

from pl_dayahead import store
from pl_dayahead.timeutils import ensure_utc

# raw series name -> column name in the hourly table
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
    """Build the hourly table from entsoe_latest rows (series, ts_utc, resolution_min, value)."""
    df = raw[raw["series"].isin(COLUMNS)].copy()
    report = {"ignored_series": sorted(set(raw["series"]) - set(COLUMNS))}

    df["ts_utc"] = ensure_utc(df["ts_utc"])
    df["hour"] = df["ts_utc"].dt.floor("h")
    df["weighted"] = df["value"] * df["resolution_min"]
    per_hour = df.groupby(["series", "hour"]).agg(
        weighted=("weighted", "sum"), minutes=("resolution_min", "sum")
    )
    per_hour["value"] = per_hour["weighted"] / per_hour["minutes"]

    # Keep only hours covered for all 60 minutes.
    complete = per_hour["minutes"] == 60
    report["dropped_hours"] = per_hour.loc[~complete].groupby(level="series").size().to_dict()
    wide = per_hour.loc[complete, "value"].unstack("series").rename(columns=COLUMNS)
    for col in COLUMNS.values():
        if col not in wide:
            wide[col] = float("nan")

    # One row per hour from the first to the last, so gaps are visible as NaN rows.
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
    """Rebuild the `hourly` table from entsoe_latest. Returns the build report."""
    raw = con.sql("SELECT series, ts_utc, resolution_min, value FROM entsoe_latest").df()
    hourly, report = build_hourly(raw)
    con.register("staged", hourly)
    try:
        con.execute("CREATE OR REPLACE TABLE hourly AS SELECT * FROM staged")
    finally:
        con.unregister("staged")
    return report


def main() -> None:
    con = store.connect()
    report = build(con)
    con.close()

    print(f"hourly: {report['n_hours']:,} hours, {report['first']} to {report['last']} (UTC)")
    if report["first_offshore"] is not None:
        print(f"offshore wind counted as zero before {report['first_offshore']}")
    print("missing hours per column:")
    for col, n in report["missing"].items():
        print(f"  {col:20s} {n:6d}")
    print("hours dropped because their rows do not cover exactly 60 minutes:", report["dropped_hours"] or "none")
    print("negative values (should be zero):", report["negative"])
    if report["ignored_series"]:
        print("raw series not used:", report["ignored_series"])


if __name__ == "__main__":
    main()