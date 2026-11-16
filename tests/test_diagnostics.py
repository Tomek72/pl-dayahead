import numpy as np
import pandas as pd

from pl_dayahead.diagnostics import historical_ranks


def test_historical_selection_is_invariant_to_future_changes_and_truncation():
    days = pd.date_range("2023-01-01", periods=150)
    values = pd.Series(np.sin(np.arange(150)), index=days)
    off = pd.Series(days.dayofweek >= 5, index=days)
    target = days[100]
    full = historical_ranks(values, off)
    truncated = historical_ranks(values.loc[:target], off.loc[:target])
    changed = values.copy()
    changed.loc[changed.index > target] = 1e6
    pd.testing.assert_series_equal(full.loc[target], truncated.loc[target])
    pd.testing.assert_series_equal(full.loc[target], historical_ranks(changed, off).loc[target])


def test_rank_uses_only_same_type_in_the_prior_calendar_window():
    days = pd.date_range("2023-01-01", periods=100)
    values = pd.Series(np.arange(100), index=days)
    off = pd.Series(days.dayofweek >= 5, index=days)
    target = days[-1]
    ref = values[(days >= target - pd.Timedelta(days=60)) & (days < target) & (off == off.loc[target])]
    result = historical_ranks(values, off).loc[target]
    assert result.percentile == 100
    assert result.reference_days == len(ref)


def test_incomplete_history_and_sparse_reference_days_are_not_ranked():
    days = pd.date_range("2023-01-01", periods=80)
    values = pd.Series(1., index=days)
    off = pd.Series(False, index=days)
    result = historical_ranks(values, off)
    assert result.percentile.iloc[:60].isna().all()
    values.iloc[10:75] = np.nan
    assert np.isnan(historical_ranks(values, off).percentile.iloc[-1])
