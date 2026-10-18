from datetime import date

import pandas as pd

from pl_dayahead.holidays_pl import easter_sunday, is_holiday, public_holidays


def test_easter_dates():
    assert easter_sunday(2023) == date(2023, 4, 9)
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


def test_moveable_holidays_follow_easter():
    days = public_holidays(2026)
    assert days[date(2026, 4, 6)] == "Easter Monday"
    assert days[date(2026, 5, 24)] == "Pentecost Sunday"
    assert days[date(2026, 6, 4)] == "Corpus Christi"


def test_christmas_eve_from_2025_only():
    assert date(2024, 12, 24) not in public_holidays(2024)
    assert date(2025, 12, 24) in public_holidays(2025)
    assert len(public_holidays(2024)) == 13
    assert len(public_holidays(2025)) == 14


def test_is_holiday_on_delivery_days():
    days = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-06", "2026-01-07", "2026-11-11"]))
    assert list(is_holiday(days)) == [True, True, False, True]