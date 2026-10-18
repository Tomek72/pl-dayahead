"""Time handling for the Polish day-ahead price project.

Conventions used everywhere in the repo:

* Timestamps are stored and joined in UTC. Naive timestamps are refused.
* Europe/Warsaw local time is used only for delivery days, the issue time and
  calendar features (local hour, weekday).
* An interval is labelled by its start and covers [start, start + freq).
* A delivery day runs from local midnight to the next local midnight, so it has
  23, 24 or 25 hours (92, 96 or 100 quarter-hours).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

LOCAL_TZ = ZoneInfo("Europe/Warsaw")
ISSUE_TIME_LOCAL = time(11, 0)  # one hour before the 12:00 SDAC gate closure


def _local(day: date, wall: time) -> datetime:
    """A Warsaw wall-clock moment. Refuses wall times skipped by the spring DST switch."""
    local = datetime.combine(day, wall, tzinfo=LOCAL_TZ)
    round_trip = local.astimezone(timezone.utc).astimezone(LOCAL_TZ)
    if round_trip.replace(tzinfo=None) != local.replace(tzinfo=None):
        raise ValueError(f"{day} {wall} does not exist in Warsaw (DST gap)")
    return local


def ensure_utc(ts) -> pd.DatetimeIndex:
    """Return a UTC DatetimeIndex. Naive input is refused: its zone cannot be known."""
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        raise ValueError("naive timestamps: attach the source's time zone at ingest")
    return idx.tz_convert("UTC")


def delivery_day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC [start, end) of a Warsaw delivery day."""
    start = _local(day, time(0)).astimezone(timezone.utc)
    end = _local(day + timedelta(days=1), time(0)).astimezone(timezone.utc)
    return pd.Timestamp(start), pd.Timestamp(end)


def delivery_index(day: date, freq: str = "h") -> pd.DatetimeIndex:
    """UTC interval starts covering one delivery day ("h" or "15min")."""
    start, end = delivery_day_bounds(day)
    return pd.date_range(start, end, freq=freq, inclusive="left", name="ts_utc")


def issue_time_utc(target_day: date, issue_local: time = ISSUE_TIME_LOCAL) -> pd.Timestamp:
    """UTC moment the forecast for target_day is made: issue_local on the day before."""
    local = _local(target_day - timedelta(days=1), issue_local)
    return pd.Timestamp(local.astimezone(timezone.utc))


def calendar_features(ts_utc) -> pd.DataFrame:
    """Local calendar columns for UTC interval starts.

    delivery_day: the Warsaw date the interval belongs to.
    local_hour: Warsaw wall-clock hour, 0-23. On the autumn DST day hour 2 occurs
        twice and on the spring day it is missing. Demand follows the wall clock,
        so this is the hour to use as a feature.
    weekday: Monday = 0 to Sunday = 6, in Warsaw time.
    """
    idx = ensure_utc(ts_utc)
    local = idx.tz_convert(LOCAL_TZ)
    return pd.DataFrame(
        {
            "delivery_day": pd.Index(local.date),
            "local_hour": local.hour,
            "weekday": local.dayofweek,  # Monday = 0
        },
        index=idx,
    )


def missing_intervals(ts_utc, freq: str = "h") -> pd.DatetimeIndex:
    """Interval starts absent between the first and last delivery day in ts_utc.

    In UTC the grid is regular, so one date_range covers every DST day correctly.
    Used by the data checks in notebooks/01_merit_order.ipynb.
    """
    idx = ensure_utc(ts_utc)
    local_days = idx.tz_convert(LOCAL_TZ).date
    start = delivery_day_bounds(min(local_days))[0]
    end = delivery_day_bounds(max(local_days))[1]
    expected = pd.date_range(start, end, freq=freq, inclusive="left", name="ts_utc")
    return expected.difference(idx)
