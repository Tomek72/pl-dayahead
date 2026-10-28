"""The DuckDB database.

The entsoe_latest table holds one row per series and time: the value from the most recent download.
Downloading a month again replaces that month's rows, so the table never holds two copies of a value.

The eua_auctions table holds one row per EU ETS primary auction of general allowances, from EEX's
reports (ingest/eex.py). Each download replaces the whole table.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from pl_dayahead.config import DB_PATH

COLUMNS = ["series", "ts_utc", "resolution_min", "value", "retrieved_at"]

SCHEMA = """
    CREATE TABLE IF NOT EXISTS entsoe_latest (
        series          VARCHAR     NOT NULL,  -- price_da, load_da_fcst, solar_da_fcst, ...
        ts_utc          TIMESTAMPTZ NOT NULL,  -- start of the hour or quarter-hour
        resolution_min  INTEGER     NOT NULL,  -- minutes the value covers: 15, 30 or 60
        value           DOUBLE      NOT NULL,
        retrieved_at    TIMESTAMPTZ NOT NULL   -- when it was downloaded
    )
"""

AUCTION_COLUMNS = ["auction_date", "result_time_utc", "auction", "price", "volume"]

AUCTION_SCHEMA = """
    CREATE TABLE IF NOT EXISTS eua_auctions (
        auction_date     DATE        NOT NULL,  -- the day of the auction, on the Warsaw (and Frankfurt) clock
        result_time_utc  TIMESTAMPTZ NOT NULL,  -- when the clearing price was set, just after the 11:00 close
        auction          VARCHAR     NOT NULL,  -- EEX's name, such as "Auction 4. Period CAP3 EU"
        price            DOUBLE      NOT NULL,  -- clearing price, EUR per tonne of CO2
        volume           DOUBLE      NOT NULL   -- allowances sold, tonnes of CO2
    )
"""


def connect(path: Path | str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the database and create its tables if they do not exist yet. Pass ":memory:" for a temporary database in tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone = 'UTC'")  # timestamps come back in UTC, never local time
    con.execute(SCHEMA)
    con.execute(AUCTION_SCHEMA)
    return con


def replace_month(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Store one month of downloaded rows, replacing earlier rows of the same series between start and end.

    The old rows are deleted and the new ones inserted together, so a failure leaves the month as it
    was. Returns the number of rows stored.
    """
    missing = set(COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"rows lack columns: {sorted(missing)}")
    if rows.empty:
        return 0
    con.register("incoming", rows[COLUMNS])
    try:
        con.begin()
        for series in rows["series"].unique():
            con.execute(
                "DELETE FROM entsoe_latest WHERE series = ? AND ts_utc >= ? AND ts_utc < ?",
                [str(series), start.to_pydatetime(), end.to_pydatetime()],
            )
        cols = ", ".join(COLUMNS)
        con.execute(f"INSERT INTO entsoe_latest ({cols}) SELECT {cols} FROM incoming")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.unregister("incoming")
    return len(rows)


def replace_auctions(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> int:
    """Replace every stored EUA auction with `rows`, in one transaction. Returns the number of rows stored."""
    missing = set(AUCTION_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"rows lack columns: {sorted(missing)}")
    con.register("incoming", rows[AUCTION_COLUMNS])
    try:
        con.begin()
        con.execute("DELETE FROM eua_auctions")
        cols = ", ".join(AUCTION_COLUMNS)
        con.execute(f"INSERT INTO eua_auctions ({cols}) SELECT {cols} FROM incoming")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.unregister("incoming")
    return len(rows)
