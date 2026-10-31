"""Tests for the EEX auction ingest, on small made-up reports and an in-memory database."""

import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pl_dayahead import store
from pl_dayahead.ingest import eex

# Date, result time on the Warsaw clock, auction name, contract, status, price, volume.
ROWS = [
    ("2024-12-16", "2024-12-16 11:00:45", "Auction 4. Period CAP3 EU", "T3PA", "successful", 63.64, 3248000),
    ("2024-07-01", "2024-07-01 11:01:10", "Auction 4. Period DE", "T3PA", "successful", 66.07, 1890500),
    ("2024-07-02", "2024-07-02 11:40:55", "EUAA Auction CAP3 EU", "EAA3", "successful", 65.10, 500000),
    ("2024-07-03", "2024-07-03 11:00:20", "Auction 4. Period CAP3 PL", "T3PA", "cancelled", np.nan, np.nan),
]


def write_report(path, rows) -> None:
    """Save rows as EEX lays out a report: titles above, the column names on the sixth row."""
    table = pd.DataFrame(
        [(pd.Timestamp(day), pd.Timestamp(result_time), *rest) for day, result_time, *rest in rows],
        columns=["Date", "Time", "Auction Name", "Contract", "Status", "Auction Price €/tCO2", "Auction Volume tCO2"],
    )
    table.to_excel(path, startrow=eex.HEADER_ROW, index=False)


def parsed(day: date, result_utc: str, price: float) -> pd.DataFrame:
    """One auction as read_report returns it."""
    return pd.DataFrame({"auction_date": [day], "result_time_utc": [pd.Timestamp(result_utc, tz="UTC")],
                         "auction": ["Auction 4. Period CAP3 EU"], "price": [price], "volume": [1000.0]})


def test_only_cleared_auctions_of_general_allowances_are_kept(tmp_path):
    write_report(tmp_path / "report.xlsx", ROWS)
    got = eex.read_report(tmp_path / "report.xlsx")
    assert list(got["auction"]) == ["Auction 4. Period CAP3 EU", "Auction 4. Period DE"]
    assert list(got["auction_date"]) == [date(2024, 12, 16), date(2024, 7, 1)]
    assert list(got["price"]) == [63.64, 66.07]


def test_result_times_are_read_on_the_warsaw_clock_and_stored_in_utc(tmp_path):
    write_report(tmp_path / "report.xlsx", ROWS)
    got = eex.read_report(tmp_path / "report.xlsx")
    assert list(got["result_time_utc"]) == [
        pd.Timestamp("2024-12-16 10:00:45", tz="UTC"),  # winter: Warsaw is UTC+1
        pd.Timestamp("2024-07-01 09:01:10", tz="UTC"),  # summer: UTC+2
    ]


def test_reports_are_read_from_the_archive_and_alone_from_2022_on(tmp_path):
    source, data = tmp_path / "source", tmp_path / "data"
    source.mkdir()
    data.mkdir()
    for year, row in [(2021, ROWS[0]), (2023, ROWS[1])]:
        write_report(source / f"{year}.xlsx", [row])
    with zipfile.ZipFile(data / "archive.zip", "w") as archive:
        for year in (2021, 2023):
            archive.write(source / f"{year}.xlsx", f"folder/emission-spot-primary-market-auction-report-{year}-data.xlsx")
    write_report(data / "emission-spot-primary-market-auction-report-2026-data.xlsx", [
        ("2026-10-22", "2026-10-22 11:00:30", "Auction 4. Period CAP3 EU", "T3PA", "successful", 84.88, 3000000),
    ])
    got = eex.read_reports(data)
    assert sorted(got["auction_date"]) == [date(2024, 7, 1), date(2026, 10, 22)]  # the 2021 report is skipped


def test_exact_duplicates_are_dropped_and_conflicting_ones_stop_the_ingest():
    one = parsed(date(2024, 7, 1), "2024-07-01 09:01", 66.07)
    assert len(eex.check(pd.concat([one, one]))) == 1
    with pytest.raises(ValueError, match="different results"):
        eex.check(pd.concat([one, parsed(date(2024, 7, 1), "2024-07-01 09:01", 67.00)]))


def test_a_result_timed_after_its_auction_day_stops_the_ingest():
    late = parsed(date(2024, 7, 1), "2024-07-01 22:30", 66.07)  # 00:30 on 2 July in Warsaw
    with pytest.raises(ValueError, match="auction day"):
        eex.check(late)


def test_a_new_download_replaces_the_stored_auctions():
    con = store.connect(":memory:")
    store.replace_auctions(con, parsed(date(2024, 7, 1), "2024-07-01 09:01", 66.07))
    store.replace_auctions(con, parsed(date(2024, 7, 1), "2024-07-01 09:01", 66.50))
    assert con.sql("SELECT auction_date, price FROM eua_auctions").fetchall() == [(date(2024, 7, 1), 66.50)]
    con.close()
