"""The DuckDB database.

The entsoe_latest table holds one row per series and time: the value from the most recent download.
Downloading a month again replaces that month's rows, so the table never holds two copies of a value.

The eua_auctions table holds one row per EU ETS primary auction of general allowances, from EEX's
reports (ingest/eex.py). Each download replaces the whole table.

The predictions table holds one row per model and hour from the backtest (backtest.py). Each run of a
model replaces that model's rows.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from pl_dayahead.config import DB_PATH

ENTSOE_COLUMNS = ["series", "ts_utc", "resolution_min", "value", "retrieved_at"]
AUCTION_COLUMNS = ["auction_date", "result_time_utc", "auction", "price", "volume"]
PREDICTION_COLUMNS = ["model", "ts_utc", "delivery_day", "issue_time_utc", "trained_from", "trained_to", "prediction",
                      "actual", "run_at"]

SCHEMA = """
    CREATE TABLE IF NOT EXISTS entsoe_latest (
        series          VARCHAR     NOT NULL,  -- price_da, load_da_fcst, solar_da_fcst, ...
        ts_utc          TIMESTAMPTZ NOT NULL,  -- start of the hour or quarter-hour
        resolution_min  INTEGER     NOT NULL,  -- minutes the value covers: 15, 30 or 60
        value           DOUBLE      NOT NULL,
        retrieved_at    TIMESTAMPTZ NOT NULL   -- when it was downloaded
    );
    CREATE TABLE IF NOT EXISTS eua_auctions (
        auction_date     DATE        NOT NULL,  -- the day of the auction, on the Warsaw (and Frankfurt) clock
        result_time_utc  TIMESTAMPTZ NOT NULL,  -- when the clearing price was set, just after the 11:00 close
        auction          VARCHAR     NOT NULL,  -- EEX's name, such as "Auction 4. Period CAP3 EU"
        price            DOUBLE      NOT NULL,  -- clearing price, EUR per tonne of CO2
        volume           DOUBLE      NOT NULL   -- allowances sold, tonnes of CO2
    );
    CREATE TABLE IF NOT EXISTS predictions (
        model           VARCHAR,      -- the model's name in models/__init__.py
        ts_utc          TIMESTAMPTZ,  -- start of the hour forecast
        delivery_day    DATE,         -- the Warsaw delivery day of that hour
        issue_time_utc  TIMESTAMPTZ,  -- 11:00 Warsaw time on the day before
        trained_from    DATE,         -- first and last day of the training window
        trained_to      DATE,
        prediction      DOUBLE,       -- EUR/MWh, empty if an input was missing
        actual          DOUBLE,       -- the price, EUR/MWh
        run_at          TIMESTAMPTZ   -- when the backtest ran
    );
"""


def connect(path: Path | str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the database and create its tables if they do not exist yet. Pass ":memory:" for a temporary database in tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET TimeZone = 'UTC'")  # timestamps come back in UTC, never local time
    con.execute(SCHEMA)
    return con


def _replace(con: duckdb.DuckDBPyConnection, table: str, columns: list[str], rows: pd.DataFrame,
             deletes: list[tuple[str, list]]) -> int:
    """Run the `deletes`, then insert `rows` into `table`, in one transaction, so a failure changes nothing.

    Returns the number of rows stored.
    """
    missing = set(columns) - set(rows.columns)
    if missing:
        raise ValueError(f"rows lack columns: {sorted(missing)}")
    con.register("incoming", rows[columns])
    try:
        con.begin()
        for statement, parameters in deletes:
            con.execute(statement, parameters)
        names = ", ".join(columns)
        con.execute(f"INSERT INTO {table} ({names}) SELECT {names} FROM incoming")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.unregister("incoming")
    return len(rows)


def replace_month(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Store one month of downloaded rows, replacing earlier rows of the same series between start and end.

    An empty download stores nothing and leaves the month as it was. Returns the number of rows stored.
    """
    if rows.empty:
        return 0
    deletes = [("DELETE FROM entsoe_latest WHERE series = ? AND ts_utc >= ? AND ts_utc < ?",
                [str(series), start.to_pydatetime(), end.to_pydatetime()]) for series in rows["series"].unique()]
    return _replace(con, "entsoe_latest", ENTSOE_COLUMNS, rows, deletes)


def replace_auctions(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> int:
    """Replace every stored EUA auction with `rows`. Returns the number of rows stored."""
    return _replace(con, "eua_auctions", AUCTION_COLUMNS, rows, [("DELETE FROM eua_auctions", [])])


def replace_predictions(con: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> int:
    """Replace the stored forecasts of the models in `rows` with `rows`. Returns the number of rows stored."""
    deletes = [("DELETE FROM predictions WHERE model = ?", [str(model)]) for model in rows["model"].unique()]
    return _replace(con, "predictions", PREDICTION_COLUMNS, rows, deletes)
