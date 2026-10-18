"""Print the slopes the linear model learned on the most recent year of data.

Run from the project root, after staging:

    uv run python -m pl_dayahead.slopes

One row per local hour, one column per coefficient of the main linear model (see
models/linear.py). Price inputs are in EUR/MWh per EUR/MWh, the load and renewables forecasts in
EUR/MWh per GW, weekday dummies in EUR/MWh relative to Wednesday, and the holiday dummy in
EUR/MWh relative to a normal day. In the hour-23 row the last price and the same-hour lag are
the same input, so their two slopes share one effect between them; read their sum.
"""

from __future__ import annotations

import pandas as pd

from pl_dayahead import store
from pl_dayahead.features import build_features
from pl_dayahead.models.linear import latest_coefficients


def main() -> None:
    con = store.connect()
    hourly = con.sql("SELECT * FROM hourly ORDER BY ts_utc").df()
    con.close()
    features, _ = build_features(hourly)
    table, first, last = latest_coefficients(features)
    print(f"Slopes of the linear model trained on {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    print("Price inputs: EUR/MWh per EUR/MWh. Forecasts: EUR/MWh per GW. Weekdays: EUR/MWh vs Wednesday.\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(2).to_string())


if __name__ == "__main__":
    main()
