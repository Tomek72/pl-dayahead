"""Publication deadlines, and the check that every model input was published in time.

The forecast for delivery day d is made at 11:00 Warsaw time on d-1. ENTSO-E does not record when
each value was published, so each series is assumed to be published exactly at its deadline in EU
Regulation 543/2013. Every deadline falls on the day before delivery, no later than 18:00 Warsaw time.

That gives one rule: an input is in time if it comes from a day before the target day, or from the
target day itself with a deadline no later than 11:00. Prices for day d are due at 13:00 on d-1, so
they never qualify. The renewables forecast for day d is due at 18:00 on d-1, so the main models use
the forecast for d-1 instead. Only the papers' setup uses day d's forecast, as the papers do, and
check() reports which inputs rely on that. The truncation tests in tests/test_features.py check the
rule against the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

from pl_dayahead.timeutils import ISSUE_TIME_LOCAL


@dataclass(frozen=True)
class Rule:
    """A publication deadline: the Warsaw time on the day before delivery by which a series is published, and its legal source."""

    published_by: time
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

# Target-day forecasts the papers use as inputs even when they are published after the forecast is
# made (Lago et al. 2021). Prices are never on this list.
PAPERS_CONVENTION = {"load_fcst", "res_fcst"}

AGGREGATIONS = {"hour", "last", "min", "max"}


@dataclass(frozen=True)
class FeatureSpec:
    """One model input, taken from a column of the hourly table.

    day_offset is the day the value comes from, relative to the target day: 0 is the target day,
    -1 the day before, and so on. agg is which value of that day is used: "hour" the same hour as
    the target hour, "last" the last hour of the day, "min" the lowest and "max" the highest.
    """

    name: str
    series: str
    day_offset: int
    agg: str = "hour"

    def __post_init__(self) -> None:
        """Reject an unknown agg value."""
        if self.agg not in AGGREGATIONS:
            raise ValueError(f"{self.name}: agg must be one of {sorted(AGGREGATIONS)}, not {self.agg!r}")


class AvailabilityError(Exception):
    """Raised when a feature uses data that was not published in time."""


def in_time(spec: FeatureSpec) -> bool:
    """Return True if the input was published by 11:00 on the day before the target day."""
    if spec.series not in RULES:
        raise AvailabilityError(f"no publication rule for series {spec.series!r}")
    if spec.day_offset < 0:
        return True  # due on an earlier day, by 18:00 at the latest
    return spec.day_offset == 0 and RULES[spec.series].published_by <= ISSUE_TIME_LOCAL


def check(specs: Iterable[FeatureSpec]) -> list[str]:
    """Check that every feature was published in time.

    A target-day forecast published later is allowed under the papers' convention, and its name is
    returned. Anything else that is late, including any price from the target day, raises
    AvailabilityError.
    """
    convention, late = [], []
    for spec in specs:
        if in_time(spec):
            continue
        if spec.day_offset == 0 and spec.series in PAPERS_CONVENTION:
            convention.append(spec.name)
        else:
            late.append(f"{spec.name} uses {spec.series} for day {spec.day_offset:+d}, published after 11:00 the day before")
    if late:
        raise AvailabilityError("; ".join(late))
    return convention
