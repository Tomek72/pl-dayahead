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

The renewables input is wind plus solar for hour h of day d-1, the latest forecast published by
11:00 on d-1. The papers' setup, LinearPapers, uses the forecast for day d instead, which is
published at 18:00 on d-1, after the forecast is made. LinearCarbon adds the price of the latest
EU emission allowance auction public at 11:00.

In the hour-23 regression, the last price of the day before and the same-hour price are the same
number. Least squares then splits their effect equally between the two coefficients, and the
forecast is unaffected.

Print the latest coefficients with: uv run python -m pl_dayahead.slopes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pl_dayahead.config import MW_PER_GW
from pl_dayahead.features import PRICE_INPUTS, model_inputs

WEEKDAYS = {"monday": 0, "tuesday": 1, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}  # Wednesday = baseline


class LinearByHour:
    """Main version: uses the renewables forecast for the day before, the latest published by 11:00."""

    name = "linear"
    renewables = "res_fcst_lag_1d"
    features = model_inputs(renewables)

    def __init__(self) -> None:
        """Start with no coefficients; fit() estimates one set per hour."""
        self.coefficients_by_hour: dict[int, pd.Series] = {}  # local hour -> coefficient of every regression input

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the regression inputs: an intercept, the six price inputs, load and renewables in GW, six weekday dummies and the holiday dummy."""
        regressors = pd.DataFrame({"intercept": 1.0}, index=X.index)
        for column in PRICE_INPUTS:
            regressors[column] = X[column].astype(float)
        regressors["load_gw"] = X["load_fcst"].astype(float) / MW_PER_GW
        regressors["res_gw"] = X[self.renewables].astype(float) / MW_PER_GW
        for name, weekday in WEEKDAYS.items():
            regressors[name] = (X["weekday"] == weekday).astype(float)
        regressors["holiday"] = X["is_holiday"].astype(float)
        return regressors

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearByHour":
        """Estimate one least-squares regression for each hour of the day from the training rows."""
        regressors = self.design(X)
        target = y.to_numpy(dtype=float)
        hours = X["local_hour"].to_numpy()
        self.coefficients_by_hour = {}
        for hour in np.unique(hours):
            in_hour = hours == hour
            # lstsq copes with a column of zeros (no holiday in the window) and with the duplicate
            # column in the hour-23 regression by splitting the effect equally.
            solution, *_ = np.linalg.lstsq(regressors.loc[in_hour].to_numpy(), target[in_hour], rcond=None)
            self.coefficients_by_hour[int(hour)] = pd.Series(solution, index=regressors.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast each row with the coefficients of its hour. Hours not seen in training get no forecast."""
        regressors = self.design(X).to_numpy()
        hours = X["local_hour"].to_numpy()
        forecast = np.full(len(X), np.nan)
        for hour, coefficients in self.coefficients_by_hour.items():
            in_hour = hours == hour
            forecast[in_hour] = regressors[in_hour] @ coefficients.to_numpy()
        return forecast

    def coefficients(self) -> pd.DataFrame:
        """Return the coefficients as a table: one row per hour, one column per input."""
        return pd.DataFrame(self.coefficients_by_hour).T.sort_index().rename_axis("hour")


class LinearPapers(LinearByHour):
    """The papers' setup: uses the renewables forecast for the delivery day, published at 18:00 on d-1.

    That is after the 11:00 forecast, so this version could not have been run in time. Its gap to
    the main version shows how much the papers' convention lowers the error.
    """

    name = "linear_papers"
    base = "linear"
    renewables = "res_fcst"
    features = model_inputs(renewables)


class LinearAsinh(LinearByHour):
    """Robustness check: the main version fitted on asinh-transformed prices, as Ziel and Weron (2018) do.

    Each training window's prices are centred on their median and divided by their median absolute
    deviation, times 1.4826 so that it matches the standard deviation of normal data. asinh(x) is
    close to x for small x and close to a logarithm for large x, and it is defined below zero, so
    spikes pull the least-squares fit much less. The target and the six price inputs are transformed
    alike, and the forecasts are transformed back into EUR/MWh.
    """

    name = "linear_asinh"
    base = "linear"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearAsinh":
        """Set the centre and scale from the training prices, then fit on the transformed scale."""
        self.centre = float(y.median())
        self.scale = 1.4826 * float((y - self.centre).abs().median())
        return super().fit(self._transform(X), np.arcsinh((y - self.centre) / self.scale))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Forecast on the transformed scale and transform back into EUR/MWh."""
        return self.centre + self.scale * np.sinh(super().predict(self._transform(X)))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with the six price inputs on the transformed scale."""
        X = X.copy()
        for column in PRICE_INPUTS:
            X[column] = np.arcsinh((X[column] - self.centre) / self.scale)
        return X


class LinearCarbon(LinearByHour):
    """Variant: the main version plus the carbon price, so 17 coefficients per hour.

    The carbon input is the clearing price of the latest EU emission allowance auction on or before
    d-2, the latest public at 11:00 on d-1, in EUR per tonne of CO2. Its slope reads as EUR/MWh per
    EUR/t: what a euro more per tonne of CO2 adds to the price, given yesterday's prices.
    """

    name = "linear_eua"
    base = "linear"
    features = [*LinearByHour.features, "eua_lag_2d"]

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the main version's regression inputs and add the carbon price."""
        regressors = super().design(X)
        regressors["eua"] = X["eua_lag_2d"].astype(float)
        return regressors


def latest_coefficients(features: pd.DataFrame, train_days: int = 365) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Fit the main model on the most recent `train_days` days with known prices. Returns the coefficients and the first and last training day."""
    inputs = LinearByHour.features
    known = features.dropna(subset=inputs + ["price"])
    last = known["delivery_day"].max()
    first = last - pd.Timedelta(days=train_days - 1)
    window = known[known["delivery_day"] >= first]
    return LinearByHour().fit(window[inputs], window["price"]).coefficients(), first, last
