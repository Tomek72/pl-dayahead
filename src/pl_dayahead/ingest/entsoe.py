"""Download ENTSO-E data for Poland into raw_entsoe, one calendar month per request.

Run from the project root:

    uv run python -m pl_dayahead.ingest.entsoe
    uv run python -m pl_dayahead.ingest.entsoe --start 2025-09-01 --end 2025-11-01 --refresh

Each month is checked before it is stored: timestamps are converted to UTC and
trimmed to the month, duplicates and gaps are counted, and the resolution of each
series is recorded. Every attempt, including failures, is written to ingest_log.

notes: explain more about the data here
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

SOURCE = "entsoe"
SETTLE = pd.Timedelta(days=7)  # a month downloaded this long after it ended is final
PAUSE_S = 0.5  # between requests; the API bans you for 10 minutes above 400 per minute
ALLOWED_RESOLUTIONS = {15, 30, 60}

# entsoe-py column names -> our series names. Unknown columns are slugified, so a new
# category in the feed is stored under a new name instead of breaking the run.
COLUMN_NAMES = {
    "Forecasted Load": "load_da_fcst",
    "Solar": "solar_da_fcst",
    "Wind Onshore": "wind_onshore_da_fcst",
    "Wind Offshore": "wind_offshore_da_fcst",
}


# --- queries --------------------------------------------------------------------


def _prices(client, start: pd.Timestamp, end: pd.Timestamp):
    # entsoe-py returns the resolution the market cleared in: hourly before
    # 1 October 2025, 15-minute from then on.
    return client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end)


def _load_forecast(client, start: pd.Timestamp, end: pd.Timestamp):
    # process_type A01 is the day-ahead forecast.
    return client.query_load_forecast(BIDDING_ZONE, start=start, end=end, process_type="A01")


def _wind_solar_forecast(client, start: pd.Timestamp, end: pd.Timestamp):
    # A01 is the day-ahead version, not the "current" forecast that keeps updating.
    return client.query_wind_and_solar_forecast(
        BIDDING_ZONE, start=start, end=end, process_type="A01"
    )


QUERIES = {
    "price_da": _prices,
    "load_da_fcst": _load_forecast,
    "wind_solar_da_fcst": _wind_solar_forecast,
}


# --- pure helpers (tested without network) -----------------------------------------


def month_chunks(start: date, end: date) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """UTC [start, end) per Warsaw calendar month, covering delivery days start to end - 1."""
    chunks = []
    first = start.replace(day=1)
    while first < end:
        nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        lo, hi = max(first, start), min(nxt, end)
        chunks.append((delivery_day_bounds(lo)[0], delivery_day_bounds(hi)[0]))
        first = nxt
    return chunks


def series_name(column) -> str:
    if column in COLUMN_NAMES:
        return COLUMN_NAMES[column]
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") + "_da_fcst"


def row_resolution(ts: pd.Series) -> pd.Series | None:
    """Minutes covered by each row, read from the spacing to its neighbours.

    A row's resolution is the smaller of the gaps to the previous and the next row,
    counting only gaps that are a valid resolution. A 15-minute row beside a missing
    stretch therefore stays 15-minute, and a row with hourly spacing on both sides is
    hourly. This works when a source changes resolution in the middle of a month, as the
    Polish forecasts did at 00:00 UTC on 13 June 2024. Returns None if no spacing in the
    series is a valid resolution.
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
) -> tuple[pd.DataFrame, dict]:
    """Turn one entsoe-py response into checked long-format rows for raw_entsoe.

    Exact duplicates are dropped and counted. A series with two different values for
    the same interval is not stored at all and is reported as a conflict, because
    picking one of them silently would hide a data problem.

    n_missing counts quarter-hours of the chunk that no row covers, so it means the
    same thing for hourly and 15-minute data.
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
    )

    report = {"n_missing": 0, "n_duplicate": 0, "conflicts": []}
    parts = []
    for name, grp in long.groupby("series", sort=True):
        grp = grp.sort_values("ts_utc")

        repeated = grp["ts_utc"].duplicated(keep=False)
        if repeated.any():
            n_values = grp[repeated].groupby("ts_utc")["value"].nunique()
            if (n_values > 1).any():
                report["conflicts"].append(f"{name}: {int((n_values > 1).sum())} intervals with different values")
                continue
            report["n_duplicate"] += int(grp["ts_utc"].duplicated().sum())
            grp = grp.drop_duplicates("ts_utc")

        if len(grp) == 1:
            resolution = pd.Series(60, index=grp.index)
        else:
            resolution = row_resolution(grp["ts_utc"])
            if resolution is None:
                report["conflicts"].append(f"{name}: no row spacing is a valid resolution")
                continue
        grp = grp.assign(resolution_min=resolution)

        # Every quarter-hour must be covered by some row: an hourly row covers four.
        quarter = pd.Timedelta(minutes=15)
        covered = pd.DatetimeIndex(
            pd.concat(
                [grp.loc[grp["resolution_min"] > 15 * k, "ts_utc"] + k * quarter
                 for k in range(max(ALLOWED_RESOLUTIONS) // 15)]
            )
        )
        # A month that has ended must be complete; the current month only up to its last value.
        last = grp["ts_utc"].iloc[-1] + pd.Timedelta(minutes=int(grp["resolution_min"].iloc[-1]))
        upper = chunk_end if chunk_end <= retrieved_at else min(chunk_end, last)
        expected = pd.date_range(chunk_start, upper, freq=quarter, inclusive="left")
        report["n_missing"] += len(expected.difference(covered))

        parts.append(grp)

    rows = pd.concat(parts) if parts else long.iloc[0:0].assign(resolution_min=[])
    rows = rows.assign(retrieved_at=retrieved_at)[store.RAW_COLUMNS].reset_index(drop=True)
    return rows, report


# --- runner -------------------------------------------------------------------------


def ingest(start: date, end: date, *, refresh: bool = False, only: list[str] | None = None) -> int:
    """Download every query for every month in [start, end). Returns the number of failed chunks."""
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError

    client = EntsoePandasClient(api_key=entsoe_token(), retry_count=3, retry_delay=10)
    con = store.connect()
    failures = 0

    for query in only or list(QUERIES):
        for lo, hi in month_chunks(start, end):
            month = lo.tz_convert(LOCAL_TZ).strftime("%Y-%m")
            if not refresh and store.chunk_is_final(con, SOURCE, query, lo, hi, SETTLE):
                print(f"{query:20s} {month}  final, skipped")
                continue

            retrieved_at = pd.Timestamp.now(tz="UTC")
            log = dict(source=SOURCE, query=query, chunk_start=lo, chunk_end=hi, retrieved_at=retrieved_at)
            try:
                raw = QUERIES[query](client, lo, hi)
            except NoMatchingDataError:
                store.log_chunk(con, **log, status="empty")
                print(f"{query:20s} {month}  empty")
                continue
            except Exception as exc:  # logged and reported, the run continues
                failures += 1
                store.log_chunk(con, **log, status="error", message=f"{type(exc).__name__}: {exc}")
                print(f"{query:20s} {month}  ERROR {type(exc).__name__}: {exc}")
                continue
            finally:
                time.sleep(PAUSE_S)

            rows, report = to_long(raw, query, lo, hi, retrieved_at)
            n_rows = store.append_raw(con, rows)
            if report["conflicts"]:
                failures += 1
                status = "conflict"
            else:
                status = "ok" if n_rows else "empty"
            store.log_chunk(
                con,
                **log,
                status=status,
                n_rows=n_rows,
                n_missing=report["n_missing"],
                n_duplicate=report["n_duplicate"],
                message="; ".join(report["conflicts"]) or None,
            )
            print(
                f"{query:20s} {month}  {status:8s} rows={n_rows} "
                f"missing={report['n_missing']} duplicates={report['n_duplicate']}"
                + (f"  {'; '.join(report['conflicts'])}" if report["conflicts"] else "")
            )

    con.close()
    return failures


def main(argv: list[str] | None = None) -> None:
    today = date.today()
    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
    parser = argparse.ArgumentParser(description="Download ENTSO-E data for Poland.")
    parser.add_argument("--start", type=date.fromisoformat, default=START_DATE, help="first delivery day")
    parser.add_argument("--end", type=date.fromisoformat, default=next_month, help="day after the last delivery day")
    parser.add_argument("--refresh", action="store_true", help="download months already marked final")
    parser.add_argument("--only", nargs="+", choices=list(QUERIES), help="run only these queries")
    args = parser.parse_args(argv)

    failures = ingest(args.start, args.end, refresh=args.refresh, only=args.only)
    if failures:
        print(f"\n{failures} chunk(s) failed or had conflicts; see ingest_log")
        sys.exit(1)


if __name__ == "__main__":
    main()