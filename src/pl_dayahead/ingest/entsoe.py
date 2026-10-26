"""Download Polish prices, load forecasts and wind and solar forecasts from ENTSO-E, one month at a time.

Run from the project root (add --help to see the options):

    uv run python -m pl_dayahead.ingest.entsoe

Each month is checked before it is stored: times are converted to UTC, rows outside the month are
dropped, exact duplicates are removed, and the minutes each row covers are recorded. The month then
replaces any earlier download of the same month in entsoe_latest.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, timedelta

import pandas as pd

from pl_dayahead import store
from pl_dayahead.config import BIDDING_ZONE, START_DATE, entsoe_token
from pl_dayahead.timeutils import LOCAL_TZ, delivery_day_bounds, ensure_utc

PAUSE_S = 0.5  # between requests; the API bans you for 10 minutes above 400 per minute
ALLOWED_RESOLUTIONS = {15, 30, 60}

# entsoe-py column names -> our series names.
COLUMN_NAMES = {
    "Forecasted Load": "load_da_fcst",
    "Solar": "solar_da_fcst",
    "Wind Onshore": "wind_onshore_da_fcst",
    "Wind Offshore": "wind_offshore_da_fcst",
}


# --- queries --------------------------------------------------------------------


def _prices(client, start: pd.Timestamp, end: pd.Timestamp):
    """Download day-ahead prices: hourly until 30 September 2025, every 15 minutes from 1 October 2025."""
    return client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end)


def _load_forecast(client, start: pd.Timestamp, end: pd.Timestamp):
    """Download the day-ahead load forecast (process type A01)."""
    return client.query_load_forecast(BIDDING_ZONE, start=start, end=end, process_type="A01")


def _wind_solar_forecast(client, start: pd.Timestamp, end: pd.Timestamp):
    """Download the day-ahead wind and solar forecast (process type A01, not the intraday updates)."""
    return client.query_wind_and_solar_forecast(BIDDING_ZONE, start=start, end=end, process_type="A01")


QUERIES = {
    "price_da": _prices,
    "load_da_fcst": _load_forecast,
    "wind_solar_da_fcst": _wind_solar_forecast,
}


# --- helpers (tested without network) -----------------------------------------------


def month_chunks(start: date, end: date) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the days from `start` up to the day before `end` into calendar months, each given as a UTC start and end."""
    chunks = []
    first = start.replace(day=1)
    while first < end:
        nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        lo, hi = max(first, start), min(nxt, end)
        chunks.append((delivery_day_bounds(lo)[0], delivery_day_bounds(hi)[0]))
        first = nxt
    return chunks


def series_name(column) -> str:
    """Return the name a downloaded column is stored under, such as "load_da_fcst". An unknown column gets a name made from its title."""
    if column in COLUMN_NAMES:
        return COLUMN_NAMES[column]
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") + "_da_fcst"


def row_resolution(ts: pd.Series) -> pd.Series | None:
    """Work out how many minutes each row covers (15, 30 or 60) from the gaps to its neighbouring rows.

    Each row gets the smaller of the gaps to the row before and the row after, counting only gaps of
    15, 30 or 60 minutes. This handles a series that switches from hourly to 15-minute values within a
    month, as the Polish forecasts did on 13 June 2024. Returns None if no gap has a valid length.
    """
    minute = pd.Timedelta(minutes=1)
    before = ts.diff() / minute
    after = -ts.diff(-1) / minute
    valid = sorted(ALLOWED_RESOLUTIONS)
    res = pd.concat([before.where(before.isin(valid)), after.where(after.isin(valid))], axis=1).min(axis=1)
    if res.isna().all():
        return None
    return res.fillna(res.mode().iloc[0]).astype(int)


def to_long(
    raw: pd.Series | pd.DataFrame,
    default_name: str,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    retrieved_at: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """Turn one month of downloaded data into rows for entsoe_latest: one row per series and time.

    Each row holds the value, the minutes it covers and the download time. Exact duplicates are
    dropped. If a series still has two values for the same time, none of that series is stored and a
    message is returned for it, because silently picking one would hide a data problem. Returns the
    rows and the list of such messages.
    """
    frame = raw.to_frame(default_name) if isinstance(raw, pd.Series) else raw.rename(columns=series_name)
    frame = frame.copy()
    frame.index = ensure_utc(frame.index)
    frame = frame[(frame.index >= chunk_start) & (frame.index < chunk_end)]

    long = (
        frame.rename_axis("ts_utc")
        .reset_index()
        .melt(id_vars="ts_utc", var_name="series", value_name="value")
        .dropna(subset=["value"])
        .drop_duplicates()
    )

    conflicts, parts = [], []
    for name, grp in long.groupby("series", sort=True):
        grp = grp.sort_values("ts_utc")
        n_repeated = int(grp["ts_utc"].duplicated().sum())
        if n_repeated:
            conflicts.append(f"{name}: {n_repeated} timestamps with more than one value")
            continue
        if len(grp) == 1:
            resolution = pd.Series(60, index=grp.index)
        else:
            resolution = row_resolution(grp["ts_utc"])
            if resolution is None:
                conflicts.append(f"{name}: no gap between rows is 15, 30 or 60 minutes")
                continue
        parts.append(grp.assign(resolution_min=resolution))

    rows = pd.concat(parts) if parts else long.iloc[0:0].assign(resolution_min=[])
    rows = rows.assign(retrieved_at=retrieved_at)[store.COLUMNS].reset_index(drop=True)
    return rows, conflicts


# --- runner -------------------------------------------------------------------------


def ingest(start: date, end: date, only: list[str] | None = None) -> int:
    """Download every series for every month from `start` up to the day before `end`, replacing earlier downloads of those months.

    Returns the number of months that failed to download or had conflicting values.
    """
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError

    client = EntsoePandasClient(api_key=entsoe_token(), retry_count=3, retry_delay=10, timeout=60)  # give up on a request after 60 s instead of waiting forever
    con = store.connect()
    failures = 0

    for query in only or list(QUERIES):
        for lo, hi in month_chunks(start, end):
            month = lo.tz_convert(LOCAL_TZ).strftime("%Y-%m")
            retrieved_at = pd.Timestamp.now(tz="UTC")
            try:
                raw = QUERIES[query](client, lo, hi)
            except NoMatchingDataError:
                print(f"{query:20s} {month}  no data")
                continue
            except Exception as exc:  # reported, and the run moves on to the next month
                failures += 1
                message = re.sub(r"securityToken=[^&\s]+", "securityToken=HIDDEN", str(exc))  # never print the token
                print(f"{query:20s} {month}  ERROR {type(exc).__name__}: {message}")
                continue
            finally:
                time.sleep(PAUSE_S)

            rows, conflicts = to_long(raw, query, lo, hi, retrieved_at)
            n_rows = store.replace_month(con, rows, lo, hi)
            if conflicts:
                failures += 1
            print(f"{query:20s} {month}  rows={n_rows}" + (f"  CONFLICT {'; '.join(conflicts)}" if conflicts else ""))

    con.close()
    return failures


def main(argv: list[str] | None = None) -> None:
    """Read the command-line options and run the download."""
    today = date.today()
    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
    parser = argparse.ArgumentParser(description="Download ENTSO-E data for Poland.")
    parser.add_argument("--start", type=date.fromisoformat, default=START_DATE, help="first delivery day")
    parser.add_argument("--end", type=date.fromisoformat, default=next_month, help="day after the last delivery day")
    parser.add_argument("--only", nargs="+", choices=list(QUERIES), help="run only these queries")
    args = parser.parse_args(argv)

    failures = ingest(args.start, args.end, only=args.only)
    if failures:
        print(f"\n{failures} month(s) failed or had conflicting values")
        sys.exit(1)


if __name__ == "__main__":
    main()
