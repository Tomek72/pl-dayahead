from datetime import date, time

import pandas as pd
import pytest

from pl_dayahead.timeutils import (
    calendar_features,
    delivery_day_bounds,
    delivery_index,
    ensure_utc,
    issue_time_utc,
    missing_intervals,
)

SPRING_2026 = date(2026, 3, 29)  # clocks go forward: 23-hour delivery day
AUTUMN_2026 = date(2026, 10, 25)  # clocks go back: 25-hour delivery day


@pytest.mark.parametrize(
    "day, hours",
    [
        (date(2026, 1, 15), 24),
        (SPRING_2026, 23),
        (AUTUMN_2026, 25),
        (date(2025, 3, 30), 23),
        (date(2025, 10, 26), 25),
    ],
)
def test_intervals_per_delivery_day(day, hours):
    assert len(delivery_index(day, "h")) == hours
    assert len(delivery_index(day, "15min")) == hours * 4


def test_bounds_winter_day():
    start, end = delivery_day_bounds(date(2026, 1, 15))
    assert start == pd.Timestamp("2026-01-14 23:00", tz="UTC")
    assert end == pd.Timestamp("2026-01-15 23:00", tz="UTC")


def test_bounds_autumn_day():
    start, end = delivery_day_bounds(AUTUMN_2026)
    assert start == pd.Timestamp("2026-10-24 22:00", tz="UTC")
    assert end == pd.Timestamp("2026-10-25 23:00", tz="UTC")


@pytest.mark.parametrize(
    "target, expected_utc",
    [
        (date(2026, 1, 15), "2026-01-14 10:00"),  # issued in CET, UTC+1
        (date(2026, 7, 15), "2026-07-14 09:00"),  # issued in CEST, UTC+2
        (SPRING_2026, "2026-03-28 10:00"),  # issued the day before the switch
        (date(2026, 3, 30), "2026-03-29 09:00"),  # issued on the switch day, after 02:00
    ],
)
def test_issue_time(target, expected_utc):
    assert issue_time_utc(target) == pd.Timestamp(expected_utc, tz="UTC")


def test_issue_time_in_dst_gap_is_refused():
    with pytest.raises(ValueError):
        issue_time_utc(date(2026, 3, 30), time(2, 30))  # 02:30 on 29 March never happens


def test_autumn_day_repeats_local_hour_2():
    cal = calendar_features(delivery_index(AUTUMN_2026))
    assert len(cal) == 25
    assert (cal["local_hour"] == 2).sum() == 2
    assert set(cal["delivery_day"]) == {AUTUMN_2026}


def test_spring_day_skips_local_hour_2():
    cal = calendar_features(delivery_index(SPRING_2026))
    assert 2 not in set(cal["local_hour"])
    assert len(cal) == 23


def test_naive_timestamps_refused():
    with pytest.raises(ValueError):
        ensure_utc(pd.date_range("2026-01-01", periods=3, freq="h"))


def test_gap_detection_across_dst():
    idx = delivery_index(AUTUMN_2026).append(delivery_index(date(2026, 10, 26)))
    assert missing_intervals(idx).empty
    gapped = idx.delete(3)
    assert list(missing_intervals(gapped)) == [idx[3]]
