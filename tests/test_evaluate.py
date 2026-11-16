from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from pl_dayahead.evaluate import (BENCHMARK, adjust_test_family, autocorrelation_diagnostic, available_variant_pairs, common_hours, daily_mae, diebold_mariano,
                                  holm_adjust, main_comparison_table, mae_chart, scores, to_markdown, variant_table)


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


def test_conflicting_actual_prices_and_duplicate_predictions_are_rejected():
    rows = predictions()
    conflicting = rows.copy()
    conflicting.loc[conflicting.model == "linear", "actual"] = 999
    with pytest.raises(ValueError, match="actual prices disagree"):
        common_hours(conflicting)
    with pytest.raises(ValueError, match="duplicate"):
        common_hours(pd.concat([rows, rows.iloc[:1]]))


def test_all_missing_model_stays_in_common_sample():
    rows = predictions()
    rows.loc[rows.model == "linear", "prediction"] = np.nan
    hours = common_hours(rows)
    assert hours.empty and "linear" in hours


def test_degenerate_comparisons_keep_family_size_without_claiming_significance():
    adjusted = adjust_test_family([0.02, np.nan, 0.5])
    assert adjusted[0] == 0.06 and np.isnan(adjusted[1]) and adjusted[2] == 1
    daily = pd.DataFrame({"a": [1., 2., 3.], "b": [1., 2., 3.]})
    table = main_comparison_table(daily, ["a", "b"])
    assert np.isnan(table["Holm-adjusted p-value"].iloc[0])
    assert not table["Significant after adjustment"].iloc[0]
    assert autocorrelation_diagnostic(daily, ["a", "b"])["Lag"].isna().all()


def test_single_model_and_short_samples_have_no_spurious_tests():
    daily = pd.DataFrame({"a": [1.], "b": [2.]})
    assert main_comparison_table(daily, ["a"]).empty
    assert np.isnan(diebold_mariano(daily, "a", "b")[1])
    assert autocorrelation_diagnostic(daily, ["a", "b"])["Lag"].isna().all()


@pytest.mark.parametrize("names", [["linear"], ["linear", "gbm"], ["linear", "gbm", "linear_asinh"]])
def test_partial_and_degenerate_reports_complete(tmp_path, monkeypatch, names):
    import duckdb
    import pl_dayahead.evaluate as evaluate
    from pl_dayahead import store

    path = tmp_path / "predictions.duckdb"
    con = store.connect(path)
    timestamps = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    rows = pd.concat([pd.DataFrame({"model": name, "ts_utc": timestamps, "actual": 100., "prediction": 101.,
                                   "delivery_day": timestamps.date, "issue_time_utc": timestamps,
                                   "trained_from": timestamps.date, "trained_to": timestamps.date,
                                   "run_at": timestamps}) for name in names])
    store.replace_predictions(con, rows)
    con.close()
    report = tmp_path / "results.md"
    monkeypatch.setattr(evaluate.store, "connect", lambda **kwargs: duckdb.connect(str(path), read_only=True))
    monkeypatch.setattr(evaluate, "RESULTS", report)
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "FIG_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "save", lambda *args: None)
    evaluate.main()
    text = report.read_text()
    assert "partial report" in text and "Missing registered main models" in text
    assert "0 available unique pairs" in text if len(names) == 1 else "1 available unique pairs" in text
    assert "No estimable autocorrelation diagnostic" in text
    if len(names) > 1:
        assert "Not testable" in text


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


def test_the_statistic_and_two_sided_p_value_match_a_hand_computed_example():
    # Daily differences 2, 0, 4, 2: mean 2, deviations 0, -2, 2, 0, and autocovariances, divided by the 4 days,
    # of 2 at lag 0 and -1 at lag 1. Newey-West with one lag: 2 + 2 x (1 - 1/2) x (-1) = 1, so DM = 2 / sqrt(1/4) = 4.
    # With no lags the variance is 2 and DM = 2 / sqrt(2/4) = 2 sqrt(2).
    daily = pd.DataFrame({"better": [10.0, 10.0, 10.0, 10.0], "worse": [12.0, 10.0, 14.0, 12.0]})
    statistic, p_value = diebold_mariano(daily, "better", "worse", lags=1)
    assert np.isclose(statistic, 4.0)
    assert np.isclose(p_value, 2 * (1 - NormalDist().cdf(4.0)))
    assert np.isclose(diebold_mariano(daily, "better", "worse", lags=0)[0], 2 * np.sqrt(2))


def test_swapping_model_order_reverses_statistic_but_keeps_two_sided_p_value():
    daily = daily_mae(year_of_forecasts(0.01, seed=1), ["naive", "good"])
    stat_ab, p_ab = diebold_mariano(daily, "good", "naive")
    stat_ba, p_ba = diebold_mariano(daily, "naive", "good")
    assert np.isclose(stat_ab, -stat_ba)
    assert np.isclose(p_ab, p_ba)


def test_identical_forecasts_are_not_tested():
    hours = year_of_forecasts(0.0)
    hours["copy"] = hours["naive"]
    assert np.isnan(diebold_mariano(daily_mae(hours, ["naive", "copy"]), "copy", "naive")[1])


def test_main_comparisons_contain_each_unique_pair_once():
    rng = np.random.default_rng(12)
    daily = pd.DataFrame({name: rng.normal(i, 1, 30) for i, name in enumerate("abcd")})
    table = main_comparison_table(daily, list("abcd"))
    pairs = list(zip(table["Model A"], table["Model B"]))
    assert len(pairs) == 6
    assert len(set(pairs)) == 6
    assert pairs == [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")]


def test_holm_step_down_adjustment_and_bounds():
    raw = np.array([0.01, 0.04, 0.03, 0.8, 0.2, 0.7])
    adjusted = holm_adjust(raw)
    expected = np.array([0.06, 0.16, 0.15, 1.0, 0.6, 1.0])
    assert np.allclose(adjusted, expected)
    assert np.all((0 <= adjusted) & (adjusted <= 1))
    # Ordered p-values must yield monotone Holm-adjusted values.
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)


def test_main_pair_adjustment_preserves_raw_p_values_and_daily_effects():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm", "naive": "linear"})
    daily = daily_mae(hours, ["gbm", "linear"])
    table = main_comparison_table(daily, ["gbm", "linear"])
    expected_p = diebold_mariano(daily, "gbm", "linear")[1]
    assert np.isclose(table.loc[0, "Raw two-sided p-value"], expected_p)
    assert np.isclose(table.loc[0, "Mean daily MAE difference (B − A)"], (daily.linear - daily.gbm).mean())


def test_a_variant_against_its_base_model():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm_papers", "naive": "gbm"})
    table = variant_table(hours, {"gbm_papers": "gbm"})
    mae = hours[["gbm", "gbm_papers"]].sub(hours["actual"], axis=0).abs().mean()
    row = table.loc["gbm_papers"]
    assert row["base"] == "gbm"
    assert np.isclose(row["difference"], mae["gbm_papers"] - mae["gbm"]) and row["difference"] < 0
    assert np.isclose(row["difference %"], 100 * row["difference"] / mae["gbm"])
    assert row["raw p-value"] < 0.001
    assert row["Holm-adjusted p-value"] < 0.001


def test_the_variant_p_value_is_two_sided():
    hours = year_of_forecasts(0.01, seed=1).rename(columns={"good": "gbm_papers", "naive": "gbm"})
    daily = daily_mae(hours, ["gbm", "gbm_papers"])
    p_value = diebold_mariano(daily, "gbm_papers", "gbm")[1]
    row = variant_table(hours, {"gbm_papers": "gbm"}).loc["gbm_papers"]
    assert np.isclose(row["raw p-value"], p_value)
    assert np.isclose(row["Holm-adjusted p-value"], p_value)  # singleton family in this unit fixture
    errors = hours[["gbm", "gbm_papers"]].sub(hours["actual"], axis=0).abs().mean()
    assert np.isclose(row["difference"], errors.gbm_papers - errors.gbm)


def test_variant_holm_adjustment_does_not_change_effect_or_raw_p_values():
    hours = year_of_forecasts(0.02, seed=2).rename(columns={"good": "linear_papers", "naive": "linear"})
    hours["linear_asinh"] = hours["actual"] + 0.9 * (hours["linear"] - hours["actual"])
    hours["linear_eua"] = hours["actual"] + 1.01 * (hours["linear"] - hours["actual"])
    variants = {"linear_papers": "linear", "linear_asinh": "linear", "linear_eua": "linear"}
    table = variant_table(hours, variants)
    daily = daily_mae(hours, list({name for pair in variants.items() for name in pair}))
    for variant, base in variants.items():
        row = table.loc[variant]
        expected_raw = diebold_mariano(daily, variant, base)[1]
        assert np.isclose(row["raw p-value"], expected_raw)
        assert np.isclose(row["difference"],
                          (hours[variant] - hours["actual"]).abs().mean()
                          - (hours[base] - hours["actual"]).abs().mean())
    assert np.all((table["Holm-adjusted p-value"] >= 0)
                  & (table["Holm-adjusted p-value"] <= 1))


def test_no_row_for_a_variant_run_without_its_base():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm_papers"})
    assert variant_table(hours, {"gbm_papers": "gbm"}).empty


def test_available_variant_pairs_identifies_missing_registered_comparisons():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm_papers", "naive": "gbm"})
    variants = {"gbm_papers": "gbm", "linear_papers": "linear"}
    available, missing = available_variant_pairs(hours, variants)
    assert available == [("gbm_papers", "gbm")]
    assert missing == [("linear_papers", "linear")]


def test_the_headline_chart_shows_each_mae_best_first_with_the_papers_setup_below_its_model():
    hours = year_of_forecasts(0.2).rename(columns={"good": "gbm", "naive": "naive_weekly"})
    hours["gbm_papers"] = hours["actual"] + 0.5 * (hours["gbm"] - hours["actual"])  # half of gbm's error
    ax = mae_chart(hours, ["naive_weekly", "gbm"]).axes[0]
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == ["Gradient-boosted trees", "papers' setup", "Weekly naive"]
    mae = hours[["gbm", "gbm_papers", "naive_weekly"]].sub(hours["actual"], axis=0).abs().mean()
    assert np.allclose([bar.get_width() for bar in ax.patches], mae)
