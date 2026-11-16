"""Download Polish prices, load forecasts and wind and solar forecasts from ENTSO-E, one month at a time,
and the prices of Germany-Luxembourg and Czechia for notebook 02's check of market coupling.

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
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import pandas as pd

from pl_dayahead import store
from pl_dayahead.config import BIDDING_ZONE, START_DATE, entsoe_token
from pl_dayahead.timeutils import LOCAL_TZ, delivery_day_bounds, ensure_utc

PAUSE_SECONDS = 0.5  # between requests; the API bans you for 10 minutes above 400 per minute
RESOLUTIONS = {"PT15M": 15, "PT30M": 30, "PT60M": 60}

# ENTSO-E production-type codes -> our series names.
PRODUCTION_TYPES = {"B16": "solar_da_fcst", "B19": "wind_onshore_da_fcst", "B18": "wind_offshore_da_fcst"}

# Series name -> download. Prices are hourly until 30 September 2025 and every 15 minutes from 1 October
# 2025. Both forecasts are the day-ahead ones (process type A01), not the intraday updates.
QUERIES = {
    "price_da": lambda client, start, end: client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end),
    "load_da_fcst": lambda client, start, end: client.query_load_forecast(
        BIDDING_ZONE, start=start, end=end, process_type="A01"),
    "wind_solar_da_fcst": lambda client, start, end: client.query_wind_and_solar_forecast(
        BIDDING_ZONE, start=start, end=end, process_type="A01"),
    # Neighbouring bidding zones, for notebook 02 only; no model uses them.
    "price_da_de_lu": lambda client, start, end: client.query_day_ahead_prices("DE_LU", start=start, end=end, sequence=1),
    "price_da_cz": lambda client, start, end: client.query_day_ahead_prices("CZ", start=start, end=end),
}


def month_chunks(start: date, end: date) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the days from `start` up to the day before `end` into calendar months, each given as a UTC start and end."""
    chunks = []
    month_start = start.replace(day=1)
    while month_start < end:
        next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        first_day, end_day = max(month_start, start), min(next_month, end)
        chunks.append((delivery_day_bounds(first_day)[0], delivery_day_bounds(end_day)[0]))
        month_start = next_month
    return chunks


def period_rows(period: ET.Element, curve: str, value_tag: str) -> pd.DataFrame:
    """Read the declared interval length and point positions, retaining gaps in fixed-block curves.

    A03 is a step curve: a value lasts until the next supplied position or the period end.
    A01 points cover exactly one interval. See ENTSO-E's curve-type guide, version 1.4, sections 4.1/4.3.
    """
    resolution = RESOLUTIONS.get(period.findtext("resolution"))
    if resolution is None or curve not in {"A01", "A03"}:
        raise ValueError("unsupported resolution or curve type")
    start, end = ensure_utc([period.findtext("timeInterval/start"), period.findtext("timeInterval/end")])
    step = pd.Timedelta(minutes=resolution)
    if end <= start or (end - start) % step:
        raise ValueError("period length is not a positive multiple of its resolution")
    points = []
    for point in period.findall("Point"):
        position = int(point.findtext("position"))
        timestamp = start + (position - 1) * step
        if not start <= timestamp < end:
            raise ValueError("point position outside its period")
        value = float(point.findtext(value_tag))
        if not pd.notna(value) or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite observation")
        points.append((timestamp, value))
    values = pd.DataFrame(points, columns=["ts_utc", "value"]).drop_duplicates()
    if values["ts_utc"].duplicated().any():
        raise ValueError("conflicting point positions")
    if curve == "A03" and not values.empty:
        values = (values.set_index("ts_utc")["value"].sort_index()
                  .reindex(pd.date_range(start, end, freq=step, inclusive="left", name="ts_utc"))
                  .ffill().dropna().reset_index())
    return values.assign(resolution_min=resolution)


def to_long(
    raw: str,
    default_name: str,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    retrieved_at: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """Parse ENTSO-E XML without inferring resolution from gaps between observations.

    Conflicting values or interval lengths reject the affected series for this chunk. Unsupported
    units, production types or curve types fail explicitly rather than changing the input meaning.
    """
    root = ET.fromstring(raw)
    for element in root.iter():
        element.tag = element.tag.split("}")[-1]  # remove the document's XML namespace
    frames, conflicts, rejected = [], [], set()
    for series in root.iter("TimeSeries"):
        name = default_name
        if default_name == "wind_solar_da_fcst":
            psr = series.findtext("MktPSRType/psrType")
            if psr not in PRODUCTION_TYPES:
                raise ValueError(f"unexpected wind/solar production type: {psr}")
            name = PRODUCTION_TYPES[psr]
            if series.find("outBiddingZone_Domain.mRID") is not None:
                raise ValueError("unexpected consumption series in wind/solar response")
        is_price = name.startswith("price_da")
        if is_price:
            if series.findtext("currency_Unit.name") != "EUR" or series.findtext("price_Measure_Unit.name") != "MWH":
                raise ValueError("expected day-ahead prices in EUR/MWh")
        elif series.findtext("quantity_Measure_Unit.name") != "MAW":
            raise ValueError("expected forecasts in MW")
        try:
            for period in series.findall("Period"):
                frame = period_rows(period, series.findtext("curveType", "A01"), "price.amount" if is_price else "quantity")
                frames.append(frame.assign(series=name))
        except (ValueError, TypeError) as error:
            conflicts.append(f"{name}: {error}")
            rejected.add(name)
    if not frames:
        return pd.DataFrame(columns=store.ENTSOE_COLUMNS), conflicts
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["ts_utc"].ge(chunk_start) & rows["ts_utc"].lt(chunk_end)].drop_duplicates()
    repeated = rows.duplicated(["series", "ts_utc"], keep=False)
    for name in rows.loc[repeated, "series"].unique():
        conflicts.append(f"{name}: conflicting values or interval lengths at the same timestamp")
        rejected.add(name)
    rows = rows[~rows["series"].isin(rejected)]
    return rows.assign(retrieved_at=retrieved_at)[store.ENTSOE_COLUMNS].reset_index(drop=True), conflicts


def ingest(start: date, end: date, only: list[str] | None = None) -> int:
    """Download every series for every month from `start` up to the day before `end`, replacing earlier downloads of those months.

    Returns the number of months that failed to download or had conflicting values.
    """
    from entsoe import EntsoeRawClient
    from entsoe.exceptions import NoMatchingDataError

    client = EntsoeRawClient(api_key=entsoe_token(), retry_count=3, retry_delay=10, timeout=60)
    con = store.connect()
    failures = 0

    for query in only or list(QUERIES):
        for month_start, month_end in month_chunks(start, end):
            month = month_start.tz_convert(LOCAL_TZ).strftime("%Y-%m")
            retrieved_at = pd.Timestamp.now(tz="UTC")
            try:
                # Match entsoe-py's boundary padding for prices; trim to the requested chunk below.
                padding = pd.Timedelta(days=1) if query.startswith("price_da") else pd.Timedelta(0)
                raw = QUERIES[query](client, month_start - padding, month_end + padding)
                rows, conflicts = to_long(raw, query, month_start, month_end, retrieved_at)
            except NoMatchingDataError:
                print(f"{query:20s} {month}  no data")
                continue
            except Exception as error:  # reported, and the run moves on to the next month
                failures += 1
                message = re.sub(r"securityToken=[^&\s]+", "securityToken=HIDDEN", str(error))  # never print the token
                print(f"{query:20s} {month}  ERROR {type(error).__name__}: {message}")
                continue
            finally:
                time.sleep(PAUSE_SECONDS)

            stored = store.replace_month(con, rows, month_start, month_end)
            if conflicts:
                failures += 1
            print(f"{query:20s} {month}  rows={stored}" + (f"  CONFLICT {'; '.join(conflicts)}" if conflicts else ""))

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
