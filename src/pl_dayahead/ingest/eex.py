"""Download EEX's results of the EU ETS primary auctions and store the clearing price of each auction.

Run from the project root (add --help to see the options):

    uv run python -m pl_dayahead.ingest.eex

EEX runs the primary auctions of EU emission allowances and publishes every result for free: one
report per year, with the years before the current one in a single archive. The reports are saved to
data/eex and read from there. Only auctions of general allowances (contract T3PA) that cleared are
kept, so aviation allowances and cancelled auctions are left out, and only years from 2022: the first
training day, 1 January 2023, needs the latest auction before it. The auctions then replace the
eua_auctions table.

Every auction's bidding window closes at 11:00, and EEX times the general-allowance results between
11:00 and 11:04 on the auction day in 2022 to 2026. A result is therefore never public at the 11:00
forecast on its own auction day. availability.py turns that into the rule for the carbon input, and
check() stops the ingest if any result is not timed on its auction day, the one fact the rule needs.
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd

from pl_dayahead import store
from pl_dayahead.config import EEX_DIR
from pl_dayahead.timeutils import LOCAL_TZ

# File name in data/eex -> download address. The current year's report moves into the archive once
# the year is over, so these two need updating each January.
REPORTS = {
    "emission-spot-primary-market-auction-report-2012-2025-data.zip":
        "https://www.eex.com/fileadmin/EEX/Downloads/Markets/Environmentals/"
        "EUA_Emission_Spot_Primary_Market_Auction_Report/Archive_Reports/"
        "emission-spot-primary-market-auction-report-2012-2025-data.zip",
    "emission-spot-primary-market-auction-report-2026-data.xlsx":
        "https://public.eex-group.com/eex/eua-auction-report/emission-spot-primary-market-auction-report-2026-data.xlsx",
}
FIRST_YEAR = 2022
HEADER_ROW = 5  # the column names are on the sixth row of every report
GENERAL_ALLOWANCES = "T3PA"  # EEX's contract code; aviation allowances are EAA3

# EEX's column names -> ours.
COLUMN_NAMES = {
    "Date": "auction_date",
    "Time": "result_time",
    "Auction Name": "auction",
    "Contract": "contract",
    "Status": "status",
    "Auction Price €/tCO2": "price",
    "Auction Volume tCO2": "volume",
}


def download(directory: Path = EEX_DIR) -> None:
    """Save every report in REPORTS to `directory`, replacing an earlier copy."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in REPORTS.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            (directory / name).write_bytes(response.read())


def report_year(name: str) -> int | None:
    """Return the year of a yearly report from its file name, or None for any other file."""
    match = re.search(r"auction-report-(\d{4})-data\.xlsx$", name)
    return int(match.group(1)) if match else None


def read_report(source) -> pd.DataFrame:
    """Read one yearly report and return its cleared auctions of general allowances, one row each.

    `source` is a path or an open file. The result time is on the Warsaw clock (EEX gives it in
    Central European time, the same clock) and is stored in UTC.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        raw = pd.read_excel(source, header=HEADER_ROW)
    missing = set(COLUMN_NAMES) - set(raw.columns)
    if missing:
        raise ValueError(f"report lacks columns: {sorted(missing)}")
    df = raw[list(COLUMN_NAMES)].rename(columns=COLUMN_NAMES).dropna(subset=["auction_date"])
    df = df[(df["contract"] == GENERAL_ALLOWANCES) & (df["status"] == "successful")]
    result_time = pd.to_datetime(df["result_time"]).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")
    return pd.DataFrame({
        "auction_date": pd.to_datetime(df["auction_date"]).dt.date,
        "result_time_utc": result_time,
        "auction": df["auction"].astype(str),
        "price": df["price"].astype(float),
        "volume": df["volume"].astype(float),
    }).reset_index(drop=True)


def read_reports(directory: Path = EEX_DIR) -> pd.DataFrame:
    """Read every yearly report from FIRST_YEAR in `directory`, whether inside an archive or on its own."""
    frames = []
    for path in sorted(directory.iterdir()):
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    year = report_year(name)
                    if year is not None and year >= FIRST_YEAR:
                        frames.append(read_report(io.BytesIO(archive.read(name))))
        else:
            year = report_year(path.name)
            if year is not None and year >= FIRST_YEAR:
                frames.append(read_report(path))
    if not frames:
        raise FileNotFoundError(f"no auction reports from {FIRST_YEAR} in {directory}")
    return pd.concat(frames, ignore_index=True)


def check(auctions: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, then stop if an auction has two different results or a result is not timed on its auction day."""
    auctions = auctions.drop_duplicates(ignore_index=True)
    twice = auctions.duplicated(["auction_date", "auction"], keep=False)
    if twice.any():
        raise ValueError(f"{twice.sum()} rows give different results for the same auction")
    result_day = auctions["result_time_utc"].dt.tz_convert(LOCAL_TZ).dt.date
    off_day = result_day != auctions["auction_date"]
    if off_day.any():
        raise ValueError(f"{off_day.sum()} results are not timed on their auction day, which availability.py assumes")
    return auctions.sort_values("result_time_utc", ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    """Read the command-line options, then download, read, check and store the auctions."""
    parser = argparse.ArgumentParser(description="Download EEX's EU ETS primary auction results.")
    parser.add_argument("--offline", action="store_true",
                        help="read the reports already in data/eex instead of downloading them")
    args = parser.parse_args(argv)

    if not args.offline:
        download()
    auctions = check(read_reports())
    con = store.connect()
    try:
        n = store.replace_auctions(con, auctions)
    finally:
        con.close()
    print(f"{n:,} auctions stored, {auctions['auction_date'].min()} to {auctions['auction_date'].max()}")


if __name__ == "__main__":
    main()
