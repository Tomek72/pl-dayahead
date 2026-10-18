import numpy as np
import pandas as pd

from pl_dayahead.evaluate import (BENCHMARK, common_hours, daily_mae, diebold_mariano, dm_table, format_p,
                                  scores, to_markdown, version_gaps)


def predictions() -> pd.DataFrame:
    ts = pd.date_range("2025-09-30 20:00", periods=6, freq="h", tz="UTC")  # crosses 1 October local
    actual = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    naive = actual + np.array([10, -10, 20, -20, 10, -10])
    model = actual + np.array([5, -5, 10, -10, 5, np.nan])  # one hour not forecast
    return pd.concat([
        pd.DataFrame({"model": "naive_weekly", "ts_utc": ts, "prediction": naive, "actual": actual}),
        pd.DataFrame({"model": "linear", "ts_utc": ts, "prediction": model, "actual": actual}),
    ])


def test_only_hours_every_model_forecast_are_scored():
    wide = common_hours(predictions())
    assert len(wide) == 5


def test_the_benchmark_is_the_weekly_naive():
    assert BENCHMARK == "naive_weekly"


def test_mae_and_relative_mae():
    wide = common_hours(predictions())
    table = scores(wide, ["naive_weekly", "linear"])
    assert table["MAE naive_weekly"].item() == 14.0  # (10+10+20+20+10)/5
    assert table["MAE linear"].item() == 7.0
    assert table["rel MAE linear"].item() == 0.5
    assert table["hours"].item() == 5


def test_period_split_at_october_2025():
    wide = common_hours(predictions())
    table = scores(wide, ["naive_weekly", "linear"], by="period")
    assert table["hours"].sum() == 5
    assert table.loc["hourly MTU (to Sep 2025)", "hours"] == 2  # 20:00 and 21:00 UTC are still 30 September in Warsaw


def test_periods_are_listed_oldest_first():
    table = scores(common_hours(predictions()), ["naive_weekly", "linear"], by="period")
    assert list(table.index) == ["hourly MTU (to Sep 2025)", "15-minute MTU (from Oct 2025)"]


def test_markdown_table():
    df = pd.DataFrame({"MAE naive": [14.0], "hours": [5]}, index=pd.Index(["all hours"], name=""))
    lines = to_markdown(df).splitlines()
    assert lines[0] == "|  | MAE naive | hours |"
    assert lines[2] == "| all hours | 14.00 | 5 |"  # hour counts stay integers


def year_of_forecasts(skill_gap: float, seed: int = 0) -> pd.DataFrame:
    """A year of hourly forecasts from two models; `good` misses by `skill_gap` less on average."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h", tz="Europe/Warsaw").tz_convert("UTC")
    actual = rng.normal(100, 30, len(ts))
    noise = rng.normal(0, 20, len(ts))
    rows = []
    for name, scale in [("naive", 1.0), ("good", 1.0 - skill_gap)]:
        rows.append(pd.DataFrame({"model": name, "ts_utc": ts, "prediction": actual + scale * noise
                                  + rng.normal(0, 5, len(ts)), "actual": actual}))
    return common_hours(pd.concat(rows))


def test_daily_mae_has_one_row_per_local_day():
    wide = year_of_forecasts(0.2)
    assert len(daily_mae(wide, ["naive", "good"])) == 365


def test_a_clearly_better_model_is_significant():
    daily = daily_mae(year_of_forecasts(0.2), ["naive", "good"])
    stat, p = diebold_mariano(daily, "good", "naive")
    assert stat > 0 and p < 0.001


def test_the_two_directions_add_up_to_one():
    daily = daily_mae(year_of_forecasts(0.01, seed=1), ["naive", "good"])
    assert np.isclose(diebold_mariano(daily, "good", "naive")[1] + diebold_mariano(daily, "naive", "good")[1], 1.0)


def test_identical_forecasts_are_not_tested():
    wide = year_of_forecasts(0.0)
    wide["copy"] = wide["naive"]
    assert np.isnan(diebold_mariano(daily_mae(wide, ["naive", "copy"]), "copy", "naive")[1])


def test_dm_table_layout():
    table = dm_table(year_of_forecasts(0.2), ["good", "naive"])
    assert table.loc["good", "naive"] < 0.001 and table.loc["naive", "good"] > 0.999
    text = format_p(table)
    assert text.loc["good", "good"] == "" and text.loc["good", "naive"] == "<0.001"


def test_version_gap_between_main_and_comparison():
    wide = year_of_forecasts(0.2).rename(columns={"good": "linear", "naive": "linear_res_d1"})
    gaps = version_gaps(wide, ["linear", "linear_res_d1"])
    mae = wide[["linear", "linear_res_d1"]].sub(wide["actual"], axis=0).abs().mean()
    row = gaps.loc["linear"]
    assert np.isclose(row["gap"], mae["linear_res_d1"] - mae["linear"])
    assert np.isclose(row["gap %"], 100 * row["gap"] / mae["linear_res_d1"])
    assert row["p main better"] == "<0.001"


def test_no_version_gap_without_both_versions():
    wide = year_of_forecasts(0.2).rename(columns={"good": "linear"})
    assert version_gaps(wide, ["linear", "naive"]).empty
