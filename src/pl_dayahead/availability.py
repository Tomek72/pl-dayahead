"""When each input becomes public, and the timing check every feature must pass.

The forecast for delivery day d is issued at 11:00 Warsaw time on d-1, one hour before the
12:00 day-ahead gate closure.

ENTSO-E does not return the time each value was published, so availability follows the
publication deadlines in the EU transparency regulation (543/2013): each value for delivery
day X is public by a fixed local time on day X-1. Brussels and Warsaw share a time zone.

There are two rules, one per kind of input:

* Prices: a hard rule. No feature may use a price that was not public at the issue time, so
  no price of day d or later ever enters an input for day d. A feature that breaks this stops
  the build.
* Day-ahead forecasts for day d: the papers' convention. The papers this project follows use
  the day-ahead forecasts for the target day as inputs and describe them as available on day
  d-1 (Lago et al. 2021). They do not compare the publication hour with the gate closure. The
  main models follow that convention. In Poland the wind and solar forecast for day d is due
  at 18:00 on d-1, seven hours after the issue time, so check() reports the features that
  rely on the convention instead of calling them public at 11:00.

The comparison models use the renewables forecast for d-1 instead, which was public at 11:00.
They declare public_at_issue_time = True, and check_model() refuses to run such a model if any
of its inputs relies on the convention.

The load forecast for day d is due by 10:00 on d-1, so it passes the rule. ENTSO-E may hold a
later update of it; the architecture document lists this under Limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import pandas as pd

from pl_dayahead.timeutils import LOCAL_TZ, issue_time_utc


@dataclass(frozen=True)
class Rule:
    published_by: time  # local time on the day before delivery
    source: str


RULES = {
    "price": Rule(
        time(13, 0),
        "Reg. 543/2013 Art. 12(2)(d): day-ahead prices no later than one hour after the 12:00 gate closure.",
    ),
    "load_fcst": Rule(
        time(10, 0),
        "Reg. 543/2013 Art. 6(2)(b): day-ahead load forecast no later than two hours before gate closure.",
    ),
    "res_fcst": Rule(
        time(18, 0),
        "Wind plus solar. Reg. 543/2013 Art. 14(2)(d): day-ahead wind and solar forecast no later than "
        "18:00 the day before delivery.",
    ),
}

# Series whose value for the target day itself may be used although it is published after the
# issue time, because the reference papers do so. Prices are never on this list.
PAPERS_CONVENTION = {
    "load_fcst": "Day-ahead forecast for day d, used as in Lago et al. (2021).",
    "res_fcst": "Day-ahead forecast for day d, used as in Lago et al. (2021).",
}

AGGREGATIONS = {"hour", "last", "min", "max"}


@dataclass(frozen=True)
class FeatureSpec:
    """One feature taken from a series in the hourly table.

    day_offset is the delivery day of the source value minus the target delivery day: 0 is
    the target day itself, -1 the day before, and so on.

    agg says which value of the source day is taken:
      "hour"  the same local hour as the target hour
      "last"  the last hour of the source day (23:00 to 24:00 local)
      "min"   the lowest value over all hours of the source day
      "max"   the highest value over all hours of the source day
    """

    name: str
    series: str
    day_offset: int
    agg: str = "hour"

    def __post_init__(self) -> None:
        if self.agg not in AGGREGATIONS:
            raise ValueError(f"{self.name}: agg must be one of {sorted(AGGREGATIONS)}, not {self.agg!r}")


class AvailabilityError(Exception):
    """A feature or model uses data it is not allowed to use."""


def available_at(series: str, delivery_day: date) -> pd.Timestamp:
    """UTC moment by which the values of `series` for `delivery_day` are public."""
    if series not in RULES:
        raise AvailabilityError(f"no availability rule for series {series!r}")
    local = datetime.combine(delivery_day - timedelta(days=1), RULES[series].published_by, tzinfo=LOCAL_TZ)
    return pd.Timestamp(local.astimezone(timezone.utc))


def check(specs: Iterable[FeatureSpec], target_days: Iterable[date]) -> list[str]:
    """Check every feature against every target day.

    A feature passes if its source values were public at the issue time on every target day.
    A feature that fails is allowed only if it takes a series on PAPERS_CONVENTION for the
    target day itself (day_offset 0); its name is returned so the caller can report it.
    Anything else raises AvailabilityError, including any late use of prices.
    """
    target_days = list(target_days)
    problems, convention = [], []
    for spec in specs:
        late = [
            t for t in target_days
            if available_at(spec.series, t + timedelta(days=spec.day_offset)) > issue_time_utc(t)
        ]
        if not late:
            continue
        if spec.series in PAPERS_CONVENTION and spec.day_offset == 0:
            convention.append(spec.name)
        else:
            problems.append(
                f"{spec.name} uses {spec.series} for day {spec.day_offset:+d}, published after "
                f"the issue time on {len(late)} of {len(target_days)} target days"
            )
    if problems:
        raise AvailabilityError("; ".join(problems))
    return convention


def check_model(name: str, features: Iterable[str], convention: Iterable[str], public_at_issue_time: bool) -> None:
    """Refuse a model that promises inputs public at 11:00 but uses one that relies on the convention."""
    if not public_at_issue_time:
        return
    used = sorted(set(features) & set(convention))
    if used:
        raise AvailabilityError(
            f"{name} must use only inputs public at the issue time, but {', '.join(used)} "
            "rely on the papers' convention"
        )
