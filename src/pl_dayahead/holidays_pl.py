"""Polish public holidays (dni ustawowo wolne od pracy).

Written out here rather than taken from a package, so the rule is visible and tested.
The list follows the Act of 18 January 1951 on public holidays, as amended; Christmas Eve
(24 December) became a public holiday from 2025.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday, by the anonymous Gregorian (Meeus/Jones/Butcher) algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def public_holidays(year: int) -> dict[date, str]:
    """Every Polish public holiday in a year, with its name."""
    easter = easter_sunday(year)
    days = {
        date(year, 1, 1): "New Year's Day",
        date(year, 1, 6): "Epiphany",
        easter: "Easter Sunday",
        easter + timedelta(days=1): "Easter Monday",
        date(year, 5, 1): "Labour Day",
        date(year, 5, 3): "Constitution Day",
        easter + timedelta(days=49): "Pentecost Sunday",
        easter + timedelta(days=60): "Corpus Christi",
        date(year, 8, 15): "Assumption of Mary",
        date(year, 11, 1): "All Saints' Day",
        date(year, 11, 11): "Independence Day",
        date(year, 12, 25): "Christmas Day",
        date(year, 12, 26): "Second Day of Christmas",
    }
    if year >= 2025:
        days[date(year, 12, 24)] = "Christmas Eve"
    return days


def is_holiday(days: pd.Series) -> pd.Series:
    """True for each delivery day (datetime64 at midnight) that is a public holiday."""
    dates = pd.to_datetime(days)
    holidays = {
        pd.Timestamp(d) for y in dates.dt.year.unique() for d in public_holidays(int(y))
    }
    return dates.isin(holidays)