"""Linear expert model: a separate least-squares regression for each hour of the day.

For each hour h of delivery day d:

    price(d,h) = b0 + b1 price(d-1,h) + b2 price(d-2,h) + b3 price(d-7,h)
               + b4 last price(d-1) + b5 min(d-1) + b6 max(d-1)
               + b7 load(d,h) + b8 renewables(h)
               + six weekday dummies + holiday dummy + error

This is the expert model of Ziel and Weron (2018, Eq. 10) with the load and renewables forecasts of
Maciejowska et al. (2022, Eq. 5), plus a holiday dummy. Forecasts are in GW, so each slope reads as
EUR/MWh per GW. Wednesday is the baseline day, so each weekday coefficient is the difference from a
Wednesday.

In the hour-23 regression, the last price of the day before and the same-hour price are the same
number. Least squares then splits their effect equally between the two coefficients, and the
forecast is unaffected.

Print the latest coefficients with: uv run python -m pl_dayahead.slopes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pl_dayahead.features import PRICE_INPUTS

WEEKDAYS = {"monday": 0, "tuesday": 1, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}  # Wednesday = baseline
MW_PER_GW = 1000.0


def _features(renewables: str) -> list[str]:
    """Return the model's input columns, using the given renewables forecast column."""
    return ["local_hour", "weekday", "is_holiday", *PRICE_INPUTS, "load_fcst", renewables]


class LinearByHour:
    """Main version: uses the renewables forecast for the delivery day, as the papers do."""

    name = "linear"
    renewables = "res_fcst"
    features = _features("res_fcst")

    def __init__(self) -> None:
        """Start with no coefficients; fit() estimates one set per hour."""
        self.coef: dict[int, pd.Series] = {}  # local hour -> coefficient of every regression input

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the regression inputs: an intercept, the six price inputs, load and renewables in GW, six weekday dummies and the holiday dummy."""
        d = pd.DataFrame({"intercept": 1.0}, index=X.index)
        for col in PRICE_INPUTS:
            d[col] = X[col].astype(float)
        d["load_gw"] = X["load_fcst"].astype(float) / MW_PER_GW
        d["res_gw"] = X[self.renewables].astype(float) / MW_PER_GW
        for name, weekday in WEEKDAYS.items():
            d[name] = (X["weekday"] == weekday).astype(float)
        d["holiday"] = X["is_holiday"].astype(float)
        return d

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearByHour":
        """Estimate one least-squares regression for each hour of the day from the training rows."""
        d = self.design(X)
        target = y.to_numpy(dtype=float)
        hours = X["local_hour"].to_numpy()
        self.coef = {}
        for hour in np.unique(hours):
            rows = hours == hour
            # lstsq copes with a column of zeros (no holiday in the window) and with the duplicate
            # column in the hour-23 regression by splitting the effect equally.
            beta, *_ = np.linalg.lstsq(d.loc[rows].to_numpy(), target[rows], rcond=None)
            self.coef[int(hour)] = pd.Series(beta, index=d.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast each row with the coefficients of its hour. Hours not seen in training get no forecast."""
        d = self.design(X).to_numpy()
        hours = X["local_hour"].to_numpy()
        out = np.full(len(X), np.nan)
        for hour, beta in self.coef.items():
            rows = hours == hour
            out[rows] = d[rows] @ beta.to_numpy()
        return out

    def coefficients(self) -> pd.DataFrame:
        """Return the coefficients as a table: one row per hour, one column per input."""
        table = pd.DataFrame(self.coef).T.sort_index()
        table.index.name = "hour"
        return table


class LinearResD1(LinearByHour):
    """Comparison version: uses the renewables forecast for the day before, so every input was published by 11:00.

    The gap to the main version shows what the delivery-day forecast is worth.
    """

    name = "linear_res_d1"
    renewables = "res_fcst_lag_1d"
    features = _features("res_fcst_lag_1d")


def latest_coefficients(features: pd.DataFrame, train_days: int = 365) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Fit the main model on the most recent `train_days` days with known prices. Returns the coefficients and the first and last training day."""
    cols = LinearByHour.features
    known = features.dropna(subset=cols + ["price"])
    last = known["delivery_day"].max()
    first = last - pd.Timedelta(days=train_days - 1)
    window = known[known["delivery_day"] >= first]
    model = LinearByHour().fit(window[cols], window["price"])
    return model.coefficients(), first, last
