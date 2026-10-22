"""Date and time helpers.

All timestamps are stored in UTC. Warsaw time is used only for delivery days, the forecast time and
calendar features such as the hour and weekday. A timestamp without a time zone is refused. Each hour
or quarter-hour is labelled by its start time. A delivery day runs from midnight to midnight Warsaw
time, so it has 23 hours or 25 during DST transitions.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

LOCAL_TZ = ZoneInfo("Europe/Warsaw")
ISSUE_TIME_LOCAL = time(11, 0)  # one hour before the 12:00 SDAC gate closure


def _local(day: date, wall: time) -> datetime:
    """Return the date and clock time as Warsaw time. Refuses times skipped by the spring DST switch."""
    local = datetime.combine(day, wall, tzinfo=LOCAL_TZ)
    round_trip = local.astimezone(timezone.utc).astimezone(LOCAL_TZ)
    if round_trip.replace(tzinfo=None) != local.replace(tzinfo=None):
        raise ValueError(f"{day} {wall} does not exist in Warsaw (DST gap)")
    return local


def ensure_utc(ts) -> pd.DatetimeIndex:
    """Convert timestamps to UTC. Refuses timestamps without a time zone"""
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        raise ValueError("naive timestamps: attach the source's time zone at ingest")
    return idx.tz_convert("UTC")


def delivery_day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the start and end of a Warsaw delivery day in UTC. The end is the start of the next day"""
    start = _local(day, time(0)).astimezone(timezone.utc)
    end = _local(day + timedelta(days=1), time(0)).astimezone(timezone.utc)
    return pd.Timestamp(start), pd.Timestamp(end)


def delivery_index(day: date, freq: str = "h") -> pd.DatetimeIndex:
    """List the UTC start time of every hour (freq="h") or quarter-hour (freq="15min") in a delivery day"""
    start, end = delivery_day_bounds(day)
    return pd.date_range(start, end, freq=freq, inclusive="left", name="ts_utc")


def issue_time_utc(target_day: date, issue_local: time = ISSUE_TIME_LOCAL) -> pd.Timestamp:
    """Return when the forecast for target_day is made: 11:00 Warsaw time the day before, in UTC.
       For 15 January that is 10:00 UTC on 14 January; in summer it is 09:00 UTC."""
    local = _local(target_day - timedelta(days=1), issue_local)
    return pd.Timestamp(local.astimezone(timezone.utc))


def calendar_features(ts_utc) -> pd.DataFrame:
    """Local calendar columns for UTC interval starts.

    delivery_day: the Warsaw date the interval belongs to.
    local_hour: Warsaw hour, 0-23. On the autumn DST day hour 2 occurs
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
    """
    List the hours (or quarter-hours) missing from a set of timestamps, between its first and last delivery day.
    """
    idx = ensure_utc(ts_utc)
    local_days = idx.tz_convert(LOCAL_TZ).date
    start = delivery_day_bounds(min(local_days))[0]
    end = delivery_day_bounds(max(local_days))[1]
    expected = pd.date_range(start, end, freq=freq, inclusive="left", name="ts_utc")
    return expected.difference(idx)
