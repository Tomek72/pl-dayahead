import pandas as pd

from pl_dayahead.holidays_pl import is_holiday


def holiday(day: str) -> bool:
    return bool(is_holiday(pd.Series(pd.to_datetime([day]))).iloc[0])


def test_fixed_date_holidays():
    assert holiday("2026-01-01") and holiday("2026-05-03") and holiday("2026-11-11")
    assert not holiday("2026-01-07")


def test_holidays_that_follow_easter():
    assert holiday("2026-04-06")  # Easter Monday
    assert holiday("2026-06-04")  # Corpus Christi
    assert not holiday("2026-04-07")


def test_christmas_eve_is_a_holiday_from_2025_only():
    assert not holiday("2024-12-24")
    assert holiday("2025-12-24")


def test_a_series_spanning_two_years():
    days = pd.Series(pd.to_datetime(["2025-12-24", "2025-12-27", "2026-01-01", "2026-01-06"]))
    assert list(is_holiday(days)) == [True, False, True, True]