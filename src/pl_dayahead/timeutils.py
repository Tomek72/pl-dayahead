"""Date and time helpers.

All timestamps are stored in UTC. Warsaw time is used only for delivery days, the forecast time and
calendar features such as the hour, the weekday and public holidays. A timestamp without a time zone
is refused. Each hour or quarter-hour is labelled by its start time. A delivery day runs from midnight
to midnight Warsaw time, so it has 23 hours or 25 on the days the clocks change.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import holidays
import pandas as pd

LOCAL_TZ = ZoneInfo("Europe/Warsaw")
ISSUE_TIME_LOCAL = time(11, 0)  # one hour before the 12:00 SDAC gate closure


def _local(day: date, clock_time: time) -> datetime:
    """Return the date and clock time as Warsaw time. Refuses a time skipped when the clocks go forward."""
    local = datetime.combine(day, clock_time, tzinfo=LOCAL_TZ)
    round_trip = local.astimezone(timezone.utc).astimezone(LOCAL_TZ)
    if round_trip.replace(tzinfo=None) != local.replace(tzinfo=None):
        raise ValueError(f"{day} {clock_time} does not exist in Warsaw (DST gap)")
    return local


def ensure_utc(timestamps) -> pd.DatetimeIndex:
    """Convert timestamps to UTC. Refuses timestamps without a time zone."""
    index = pd.DatetimeIndex(timestamps)
    if index.tz is None:
        raise ValueError("naive timestamps: attach the source's time zone at ingest")
    return index.tz_convert("UTC")


def delivery_day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the start and end of a Warsaw delivery day in UTC. The end is the start of the next day."""
    start = _local(day, time(0)).astimezone(timezone.utc)
    end = _local(day + timedelta(days=1), time(0)).astimezone(timezone.utc)
    return pd.Timestamp(start), pd.Timestamp(end)


def delivery_index(day: date, freq: str = "h") -> pd.DatetimeIndex:
    """List the UTC start time of every hour (freq="h") or quarter-hour (freq="15min") in a delivery day."""
    start, end = delivery_day_bounds(day)
    return pd.date_range(start, end, freq=freq, inclusive="left", name="ts_utc")


def issue_time_utc(target_day: date, issue_local: time = ISSUE_TIME_LOCAL) -> pd.Timestamp:
    """Return when the forecast for target_day is made, 11:00 Warsaw time the day before, in UTC.

    For 15 January that is 10:00 UTC on 14 January; in summer it is 09:00 UTC.
    """
    local = _local(target_day - timedelta(days=1), issue_local)
    return pd.Timestamp(local.astimezone(timezone.utc))


def calendar_features(ts_utc) -> pd.DataFrame:
    """Local calendar columns for UTC interval starts.

    delivery_day: the Warsaw date the interval belongs to.
    local_hour: Warsaw hour, 0-23. On the autumn clock change hour 2 occurs twice, and on the spring
        one it is missing. Demand follows the wall clock, so this is the hour to use as a feature.
    weekday: Monday = 0 to Sunday = 6, in Warsaw time.
    """
    index = ensure_utc(ts_utc)
    local = index.tz_convert(LOCAL_TZ)
    return pd.DataFrame({"delivery_day": pd.Index(local.date), "local_hour": local.hour, "weekday": local.dayofweek},
                        index=index)


def is_holiday(days: pd.Series) -> pd.Series:
    """Return True for each delivery day that is a Polish public holiday, from the holidays package."""
    dates = pd.to_datetime(days)
    poland = holidays.Poland(years=dates.dt.year.unique().tolist())
    return dates.dt.date.isin(list(poland))


def missing_intervals(ts_utc, freq: str = "h") -> pd.DatetimeIndex:
    """List the hours (or quarter-hours) missing from a set of timestamps, between its first and last delivery day."""
    index = ensure_utc(ts_utc)
    local_days = index.tz_convert(LOCAL_TZ).date
    start = delivery_day_bounds(min(local_days))[0]
    end = delivery_day_bounds(max(local_days))[1]
    return pd.date_range(start, end, freq=freq, inclusive="left", name="ts_utc").difference(index)


def day_label(day, weekday: bool = True) -> str:
    """A day as the charts, tables and notebooks write it: 'Wed 5 Mar 2025', or '5 Mar 2025' without the weekday."""
    return (f"{day:%a} " if weekday else "") + f"{day.day} {day:%b %Y}"
