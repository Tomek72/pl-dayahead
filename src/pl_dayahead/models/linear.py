"""Linear expert model: one ordinary least squares regression for each local hour of the day.

This is the expert model of Ziel and Weron (2018, Eq. 10) with the two outside inputs of
Maciejowska, Uniejewski and Weron (2022, Eq. 5), plus a holiday dummy. For target day d and
local hour h:

    price(d,h) = b0 + b1 price(d-1,h) + b2 price(d-2,h) + b3 price(d-7,h)
               + b4 last price(d-1) + b5 min(d-1) + b6 max(d-1)
               + b7 load(d,h) + b8 renewables(h)
               + six weekday dummies + holiday dummy + error

Sixteen coefficients per hour, estimated by OLS on the training window, separately for each
hour as in both papers. The same input moves the price differently at different times: extra
load at 19:00, when expensive gas units run, moves the price more than extra load at 04:00.

The inputs (see features.py):

* price lags at d-1, d-2 and d-7, and the last price, minimum and maximum of d-1. They carry
  the current price level, which follows the fuel and carbon costs the model is not given.
* load and renewables (wind plus solar) forecasts in GW, so each slope reads as EUR/MWh per GW.
  The main model uses the renewables forecast for day d, as the papers do; LinearResD1 uses
  the forecast for d-1, which was public at 11:00.
* weekday dummies. Ziel and Weron write an intercept plus all seven; that set is collinear,
  because the seven dummies always add up to 1, the same as the intercept. Dropping one gives
  exactly the same forecasts. Wednesday is the dropped day, so each weekday coefficient reads as
  the difference from an ordinary midweek day.
* a holiday dummy for Polish public holidays. This is not in the papers; the architecture
  document lists it as a deviation.

Collinearity in the last hour: in the hour-23 equation, the last price of d-1 and the
same-hour lag are the same number (Ziel and Weron note this). Least squares gives the
minimum-norm solution, and the forecast is the same as with either column alone.

Reading the slopes: each is the effect of that input with all other inputs held fixed,
including yesterday's prices. Yesterday's price already contains yesterday's weather, which
resembles today's, so part of each forecast's effect is absorbed by the price inputs. Read the
signs and how they change across hours, not the exact sizes.

Print the slopes the model learned on the latest year of data with:

    uv run python -m pl_dayahead.slopes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pl_dayahead.features import PRICE_INPUTS

WEEKDAYS = {"monday": 0, "tuesday": 1, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}  # Wednesday = baseline
MW_PER_GW = 1000.0


def _features(renewables: str) -> list[str]:
    return ["local_hour", "weekday", "is_holiday", *PRICE_INPUTS, "load_fcst", renewables]


class LinearByHour:
    name = "linear"
    renewables = "res_fcst"  # forecast for day d: the papers' convention
    features = _features("res_fcst")
    public_at_issue_time = False

    def __init__(self) -> None:
        self.coef: dict[int, pd.Series] = {}  # local hour -> slope of every design column

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """The regression inputs: intercept, price inputs, forecasts in GW, weekday and holiday dummies."""
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
        d = self.design(X)
        target = y.to_numpy(dtype=float)
        hours = X["local_hour"].to_numpy()
        self.coef = {}
        for hour in np.unique(hours):
            rows = hours == hour
            # Least squares. lstsq copes with a column of zeros (no holiday in the window) and with
            # the duplicate column in the hour-23 equation by returning the minimum-norm solution.
            beta, *_ = np.linalg.lstsq(d.loc[rows].to_numpy(), target[rows], rcond=None)
            self.coef[int(hour)] = pd.Series(beta, index=d.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        d = self.design(X).to_numpy()
        hours = X["local_hour"].to_numpy()
        out = np.full(len(X), np.nan)  # an hour never seen in training gets no forecast
        for hour, beta in self.coef.items():
            rows = hours == hour
            out[rows] = d[rows] @ beta.to_numpy()
        return out

    def coefficients(self) -> pd.DataFrame:
        """One row per local hour, one column per design input."""
        table = pd.DataFrame(self.coef).T.sort_index()
        table.index.name = "hour"
        return table


class LinearResD1(LinearByHour):
    """Comparison version: the renewables forecast for d-1 instead of d, so every input was public at 11:00.

    The gap between this model's error and LinearByHour's is what the papers' convention of
    using the day-d forecast, published at 18:00 on d-1, is worth.
    """

    name = "linear_res_d1"
    renewables = "res_fcst_lag_1d"
    features = _features("res_fcst_lag_1d")
    public_at_issue_time = True


def latest_coefficients(features: pd.DataFrame, train_days: int = 365) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Fit the main model on the last `train_days` delivery days that have prices; return the slopes and the window."""
    cols = LinearByHour.features
    known = features.dropna(subset=cols + ["price"])
    last = known["delivery_day"].max()
    first = last - pd.Timedelta(days=train_days - 1)
    window = known[known["delivery_day"] >= first]
    model = LinearByHour().fit(window[cols], window["price"])
    return model.coefficients(), first, last
