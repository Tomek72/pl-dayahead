"""Historical reference ranks for the tight-day diagnostic, independent of realised errors."""

import numpy as np
import pandas as pd


def historical_ranks(values: pd.Series, day_type: pd.Series, lookback_days: int = 60,
                     minimum_days: int = 10) -> pd.DataFrame:
    """Rank each daily value against earlier days of the same type in a fixed calendar window.

    Require a full lookback span and at least `minimum_days` observed reference values. Percentiles
    are the percentage of reference values strictly below the target, as in the retrospective check.
    Neither future observations nor realised forecast errors participate in selection.
    """
    if lookback_days < 1 or minimum_days < 1:
        raise ValueError("lookback and minimum reference count must be positive")
    if not isinstance(values.index, pd.DatetimeIndex) or not values.index.is_unique:
        raise ValueError("values require one row per delivery day")
    values = values.sort_index()
    types = day_type.reindex(values.index)
    if types.isna().any():
        raise ValueError("every day needs a day type")
    result = pd.DataFrame({"percentile": np.nan, "reference_days": 0}, index=values.index)
    if values.empty:
        return result
    window = pd.Timedelta(days=lookback_days)
    for day, value in values.items():
        if pd.isna(value) or day - window < values.index.min():
            continue
        reference = values[(values.index >= day - window) & (values.index < day)
                           & (types == types.loc[day])].dropna()
        result.loc[day, "reference_days"] = len(reference)
        if len(reference) >= minimum_days:
            result.loc[day, "percentile"] = 100 * (reference < value).mean()
    return result
