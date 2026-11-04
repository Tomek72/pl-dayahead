from statistics import NormalDist

import numpy as np
import pandas as pd

from pl_dayahead.evaluate import (BENCHMARK, common_hours, daily_mae, diebold_mariano, dm_table, format_p, mae_chart,
                                  scores, to_markdown, variant_table)


def predictions() -> pd.DataFrame:
    timestamps = pd.date_range("2025-09-30 20:00", periods=6, freq="h", tz="UTC")  # crosses 1 October local
    actual = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    naive = actual + np.array([10, -10, 20, -20, 10, -10])
    linear = actual + np.array([5, -5, 10, -10, 5, np.nan])  # one hour not forecast
    return pd.concat([
        pd.DataFrame({"model": "naive_weekly", "ts_utc": timestamps, "prediction": naive, "actual": actual}),
        pd.DataFrame({"model": "linear", "ts_utc": timestamps, "prediction": linear, "actual": actual}),
    ])


def test_only_hours_every_model_forecast_are_scored():
    hours = common_hours(predictions())
    assert len(hours) == 5


def test_the_benchmark_is_the_weekly_naive():
    assert BENCHMARK == "naive_weekly"


def test_mae_and_relative_mae():
    hours = common_hours(predictions())
    table = scores(hours, ["naive_weekly", "linear"])
    assert table["MAE naive_weekly"].item() == 14.0  # (10+10+20+20+10)/5
    assert table["MAE linear"].item() == 7.0
    assert table["rel MAE linear"].item() == 0.5
    assert table["hours"].item() == 5


def test_period_split_at_october_2025():
    hours = common_hours(predictions())
    table = scores(hours, ["naive_weekly", "linear"], by="period")
    assert table["hours"].sum() == 5
    assert table.loc["hourly MTU (to Sep 2025)", "hours"] == 2  # 20:00 and 21:00 UTC are still 30 September in Warsaw


def test_periods_are_listed_oldest_first():
    table = scores(common_hours(predictions()), ["naive_weekly", "linear"], by="period")
    assert list(table.index) == ["hourly MTU (to Sep 2025)", "15-minute MTU (from Oct 2025)"]


def test_markdown_table():
    table = pd.DataFrame({"MAE naive": [14.0], "hours": [5]}, index=pd.Index(["all hours"], name=""))
    lines = to_markdown(table).splitlines()
    assert lines[0] == "|  | MAE naive | hours |"
    assert lines[2] == "| all hours | 14.00 | 5 |"  # hour counts stay integers


def year_of_forecasts(skill_gap: float, seed: int = 0) -> pd.DataFrame:
    """A year of hourly forecasts from two models; `good` misses by `skill_gap` less on average."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01 00:00", "2025-12-31 23:00", freq="h", tz="Europe/Warsaw").tz_convert("UTC")
    actual = rng.normal(100, 30, len(timestamps))
    noise = rng.normal(0, 20, len(timestamps))
    rows = []
    for name, scale in [("naive", 1.0), ("good", 1.0 - skill_gap)]:
        rows.append(pd.DataFrame({"model": name, "ts_utc": timestamps, "prediction": actual + scale * noise
                                  + rng.normal(0, 5, len(timestamps)), "actual": actual}))
    return common_hours(pd.concat(rows))


def test_daily_mae_has_one_row_per_local_day():
    hours = year_of_forecasts(0.2)
    assert len(daily_mae(hours, ["naive", "good"])) == 365


def test_a_clearly_better_model_is_significant():
    daily = daily_mae(year_of_forecasts(0.2), ["naive", "good"])
    stat, p = diebold_mariano(daily, "good", "naive")
    assert stat > 0 and p < 0.001


def test_the_statistic_matches_a_hand_computed_example():
    # Daily differences 2, 0, 4, 2: mean 2, deviations 0, -2, 2, 0, and autocovariances, divided by the 4 days,
    # of 2 at lag 0 and -1 at lag 1. Newey-West with one lag: 2 + 2 x (1 - 1/2) x (-1) = 1, so DM = 2 / sqrt(1/4) = 4.
    # With no lags the variance is 2 and DM = 2 / sqrt(2/4) = 2 sqrt(2).
    daily = pd.DataFrame({"better": [10.0, 10.0, 10.0, 10.0], "worse": [12.0, 10.0, 14.0, 12.0]})
    statistic, p_value = diebold_mariano(daily, "better", "worse", lags=1)
    assert np.isclose(statistic, 4.0)
    assert np.isclose(p_value, 1 - NormalDist().cdf(4.0))
    assert np.isclose(diebold_mariano(daily, "better", "worse", lags=0)[0], 2 * np.sqrt(2))


def test_the_two_directions_add_up_to_one():
    daily = daily_mae(year_of_forecasts(0.01, seed=1), ["naive", "good"])
    assert np.isclose(diebold_mariano(daily, "good", "naive")[1] + diebold_mariano(daily, "naive", "good")[1], 1.0)


def test_identical_forecasts_are_not_tested():
    hours = year_of_forecasts(0.0)
    hours["copy"] = hours["naive"]
    assert np.isnan(diebold_mariano(daily_mae(hours, ["naive", "copy"]), "copy", "naive")[1])


def test_dm_table_layout():
    table = dm_table(year_of_forecasts(0.2), ["good", "naive"])
    assert table.loc["good", "naive"] < 0.001 and table.loc["naive", "good"] > 0.999
    text = table.map(format_p)
    assert text.loc["good", "good"] == "" and text.loc["good", "naive"] == "<0.001"


def test_a_variant_against_its_base_model():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm_papers", "naive": "gbm"})
    table = variant_table(hours, {"gbm_papers": "gbm"})
    mae = hours[["gbm", "gbm_papers"]].sub(hours["actual"], axis=0).abs().mean()
    row = table.loc["gbm_papers"]
    assert row["base"] == "gbm"
    assert np.isclose(row["difference"], mae["gbm_papers"] - mae["gbm"]) and row["difference"] < 0
    assert np.isclose(row["difference %"], 100 * row["difference"] / mae["gbm"])
    assert row["p"] == "<0.001"


def test_the_variant_p_value_is_two_sided():
    hours = year_of_forecasts(0.01, seed=1).rename(columns={"good": "gbm_papers", "naive": "gbm"})
    one_sided = diebold_mariano(daily_mae(hours, ["gbm", "gbm_papers"]), "gbm_papers", "gbm")[1]
    p_value = variant_table(hours, {"gbm_papers": "gbm"}).loc["gbm_papers", "p"]
    assert p_value == format_p(2 * min(one_sided, 1 - one_sided))


def test_no_row_for_a_variant_run_without_its_base():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm_papers"})
    assert variant_table(hours, {"gbm_papers": "gbm"}).empty


def test_the_headline_chart_shows_each_mae_best_first_with_the_papers_setup_below_its_model():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm", "naive": "naive_weekly"})
    hours["gbm_papers"] = hours["actual"] + 0.5 * (hours["gbm"] - hours["actual"])  # half of gbm's error
    ax = mae_chart(hours, ["naive_weekly", "gbm"]).axes[0]
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["Gradient-boosted trees", "papers' setup", "Weekly naive"]
    mae = hours[["gbm", "gbm_papers", "naive_weekly"]].sub(hours["actual"], axis=0).abs().mean()
    assert np.allclose([bar.get_width() for bar in ax.patches], mae)
