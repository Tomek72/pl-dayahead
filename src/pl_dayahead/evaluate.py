"""Compare models on exactly the same hours.

Run from the project root, after one or more backtests:

    uv run python -m pl_dayahead.evaluate

Every model is scored only on hours where all models made a forecast and the actual price
is known, so no model looks better by skipping hard hours.

Metrics:

* MAE, mean absolute error in EUR/MWh: the average size of the miss. It is used rather than
  squared error to reduce the influence of extreme hours (spikes, deep negative prices).
* Relative MAE: a model's MAE divided by the benchmark's MAE on the same hours. The benchmark
  is the weekly naive forecast ("naive_weekly", see models/naive.py), as recommended by Lago et
  al. (2021). Below 1 means the model beats it; 0.8 means its errors are 20% smaller. Dividing
  by the benchmark also makes periods comparable: in a volatile month every model misses by
  more, but the ratio still shows skill.

The results are broken down by period (before and after the move to 15-minute products on
1 October 2025), by year, by month and by local hour, and written to reports/results.md. The
README's headline chart, each main model's MAE with the papers' setup below gbm and the linear
model, is written to reports/figures/mae_by_model.png.

Only models registered in models/__init__.py are scored. Rows left in the predictions table
by a model that no longer exists (an old name, say) are listed and ignored.

Variants. Some registered models are variants of a main model (see models/__init__.py): the
papers' setup, which uses the renewables forecast for day d, assumed available at its 18:00 deadline on d-1, after the
issue time; two robustness checks; and the carbon variants, which add the price of the latest EU
emission allowance auction public at 11:00. They are scored on the same hours as every other model
but kept out of the main tables. A separate table compares each with its base model: the
difference in MAE, and a two-sided Diebold-Mariano p-value, the chance of a difference this
large if the two were equally accurate.

Is a difference real or luck? A lower MAE over 24,000 hours can still come from a handful of
lucky weeks. The Diebold-Mariano test asks whether the mean difference between two models' daily
errors differs from zero:

1. For each delivery day, take each model's MAE over that day's hours. Hours within one day
   share the same weather and fuel news, so their errors are not independent; days are the
   unit of evidence (about 1,000 of them), not hours.
2. Take the daily difference between the two models and test whether its average differs from
   zero, like a two-sided t-test on the differences.
3. Consecutive days are also alike (a volatile week is volatile all week), so the standard
   error uses a Newey-West correction over 7 days of lags. Its effect on the standard error
   depends on the autocovariances; it need not make every comparison more conservative.

The daily mean absolute error is Lago et al.'s daily loss (the sum of absolute errors over the
day's hours, the L1 norm) divided by the number of hours. Dividing every day by 24 leaves the
test statistic unchanged; only the 23- and 25-hour clock-change days are scaled slightly
differently.

Main models are compared once per unique pair with two-sided tests; Holm adjustment is applied
across the six comparisons. Variant tests form a separate family over the reported comparisons and
receive their own Holm adjustment. The asymptotic p-values require a stable, sufficiently weakly dependent
daily loss-difference process for the Newey-West estimate and normal approximation to be reliable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pl_dayahead import store
from pl_dayahead.config import FIG_DIR, PROJECT_ROOT, SDAC_15MIN_START
from pl_dayahead.models import MODELS, PAPERS_SETUP, VARIANTS
from pl_dayahead.style import BASELINE, GRID, INK, INK_2, MODEL_COLOURS, MUTED, SOURCE_BACKTEST, SURFACE, save
from pl_dayahead.timeutils import LOCAL_TZ, day_label

BENCHMARK = "naive_weekly"
SIMILAR_DAY = "naive_similar_day"  # the stronger benchmark, also reported as a percentage change
RESULTS = PROJECT_ROOT / "reports" / "results.md"
CHART = "mae_by_model.png"
PERIODS = ["hourly MTU (to Sep 2025)", "15-minute MTU (from Oct 2025)"]
DM_LAGS = 7  # primary Newey-West bandwidth in delivery days
DM_BANDWIDTHS = (7, 14, 28)
ALPHA = 0.05
MODEL_LABELS = {"gbm": "Gradient-boosted trees", "linear": "Linear expert model",
                "naive_similar_day": "Similar-day naive", "naive_weekly": "Weekly naive"}


def common_hours(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per hour, one prediction column per model; only hours every model forecast."""
    if predictions.empty:
        raise ValueError("no predictions to evaluate")
    if predictions[["model", "ts_utc"]].isna().any().any():
        raise ValueError("predictions require a model and timestamp")
    if predictions.duplicated(["model", "ts_utc"]).any():
        raise ValueError("duplicate model/hour predictions")
    if predictions.groupby("ts_utc")["actual"].nunique(dropna=False).gt(1).any():
        raise ValueError("actual prices disagree across models; rerun against a common data snapshot")
    if predictions[["prediction", "actual"]].isin([np.inf, -np.inf]).any().any():
        raise ValueError("non-finite prediction or actual price")
    hours = predictions.pivot(index="ts_utc", columns="model", values="prediction").sort_index()
    actual = predictions.groupby("ts_utc")["actual"].first()
    hours = hours.join(actual).dropna()
    local = hours.index.tz_convert(LOCAL_TZ)
    after = local.date >= SDAC_15MIN_START
    # An ordered category, so tables list the older period first.
    hours["period"] = pd.Categorical.from_codes(after.astype(int), categories=PERIODS, ordered=True)
    hours["year"] = local.year
    hours["month"] = local.strftime("%Y-%m")
    hours["local_hour"] = local.hour
    return hours


def scores(hours: pd.DataFrame, models: list[str], by: str | None = None) -> pd.DataFrame:
    """MAE per model, and relative MAE against the benchmark, overall or per group."""
    errors = hours[models].sub(hours["actual"], axis=0).abs()
    if by is not None:
        errors[by] = hours[by]
        groups = errors.groupby(by, observed=True)  # observed: skip a period with no hours yet
        mae = groups[models].mean()
        counts = groups.size()
    else:
        mae = errors[models].mean().to_frame("all hours").T
        counts = pd.Series([len(errors)], index=mae.index)
    table = mae.add_prefix("MAE ")
    if BENCHMARK in models:
        for model in models:
            if model != BENCHMARK:
                table[f"rel MAE {model}"] = mae[model] / mae[BENCHMARK]
    table["hours"] = counts
    return table


def daily_mae(hours: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per local delivery day, each model's MAE over that day's hours."""
    day = hours.index.tz_convert(LOCAL_TZ).date
    return hours[models].sub(hours["actual"], axis=0).abs().groupby(day).mean()


def diebold_mariano(daily: pd.DataFrame, model_a: str, model_b: str, lags: int = DM_LAGS) -> tuple[float, float]:
    """Two-sided test of equal daily MAE; positive differences favour model A. Returns statistic and p-value."""
    difference = (daily[model_b] - daily[model_a]).to_numpy()  # positive when A has lower daily MAE
    days = len(difference)
    if days < 2:
        return math.nan, math.nan
    centred = difference - difference.mean()
    variance = centred @ centred / days
    for lag in range(1, min(lags, days - 1) + 1):  # Newey-West: add the covariance with `lag` days earlier
        variance += 2 * (1 - lag / (lags + 1)) * (centred[lag:] @ centred[:-lag]) / days
    if variance <= 0:
        return math.nan, math.nan  # zero/negative estimated variance: no valid asymptotic test
    statistic = difference.mean() / math.sqrt(variance / days)
    return statistic, math.erfc(abs(statistic) / math.sqrt(2))  # two-sided standard-normal tail probability


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a finite one-dimensional sequence between zero and one")
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def format_probability(p_value: float) -> str:
    """Round-trip representation of a full-precision probability for generated reports."""
    return "Not testable" if math.isnan(p_value) else repr(float(p_value))


def adjust_test_family(p_values: Sequence[float]) -> np.ndarray:
    """Keep untestable comparisons in the family as p=1 for adjustment, but report them as missing."""
    raw = np.asarray(p_values, dtype=float)
    missing = np.isnan(raw)
    adjusted = holm_adjust(np.where(missing, 1.0, raw))
    adjusted[missing] = np.nan
    return adjusted


def main_comparison_table(daily: pd.DataFrame, models: list[str], lags: int = DM_LAGS) -> pd.DataFrame:
    """One two-sided DM test per unique main-model pair, with a family-wise Holm adjustment."""
    rows = []
    for model_a, model_b in combinations(models, 2):
        _, p_raw = diebold_mariano(daily, model_a, model_b, lags=lags)
        rows.append({
            "Model A": model_a,
            "Model B": model_b,
            "Mean daily MAE difference (B − A)": (daily[model_b] - daily[model_a]).mean(),
            "Raw two-sided p-value": p_raw,
        })
    table = pd.DataFrame(rows, columns=["Model A", "Model B", "Mean daily MAE difference (B − A)", "Raw two-sided p-value"])
    adjusted = adjust_test_family(table["Raw two-sided p-value"].to_numpy())
    table["Holm-adjusted p-value"] = adjusted
    table["Significant after adjustment"] = adjusted <= ALPHA
    return table


def autocorrelation_diagnostic(daily: pd.DataFrame, models: list[str], first_lag: int = 8,
                               last_lag: int = 28) -> pd.DataFrame:
    """Largest absolute sample ACF beyond the primary HAC bandwidth for each unique pair."""
    rows = []
    for model_a, model_b in combinations(models, 2):
        difference = (daily[model_b] - daily[model_a]).to_numpy(dtype=float)
        centred = difference - difference.mean()
        denominator = centred @ centred
        acf = np.array([(centred[lag:] @ centred[:-lag]) / denominator
                        for lag in range(first_lag, min(last_lag, len(centred) - 1) + 1)]) if denominator > 0 else np.array([])
        max_position = int(np.argmax(np.abs(acf))) if len(acf) else None
        rows.append({
            "Model A": model_a,
            "Model B": model_b,
            "Largest absolute ACF, lags 8–28": acf[max_position] if max_position is not None else np.nan,
            "Lag": first_lag + max_position if max_position is not None else np.nan,
        })
    return pd.DataFrame(rows, columns=["Model A", "Model B", "Largest absolute ACF, lags 8–28", "Lag"])


def bandwidth_sensitivity_table(daily: pd.DataFrame, models: list[str],
                                bandwidths: Sequence[int] = DM_BANDWIDTHS) -> pd.DataFrame:
    """Two-sided raw DM p-values for primary-model pairs across HAC bandwidths."""
    rows = []
    for model_a, model_b in combinations(models, 2):
        row = {"Model A": model_a, "Model B": model_b}
        for bandwidth in bandwidths:
            row[f"Raw p, {bandwidth} lags"] = diebold_mariano(daily, model_a, model_b, lags=bandwidth)[1]
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model A", "Model B", *[f"Raw p, {bandwidth} lags" for bandwidth in bandwidths]])


def format_report_p_values(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Copy a table and render selected full-precision p-value columns for Markdown output."""
    report = table.copy()
    for column in columns:
        report[column] = report[column].map(format_probability)
    return report


def available_variant_pairs(
        hours: pd.DataFrame, variants: dict[str, str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return registered variant/base pairs available in the common sample and those missing from it."""
    available, missing = [], []
    for variant, base in variants.items():
        (available if variant in hours and base in hours else missing).append((variant, base))
    return available, missing


def variant_table(hours: pd.DataFrame, variants: dict[str, str], lags: int = DM_LAGS) -> pd.DataFrame:
    """Exploratory variant comparisons with a separate Holm adjustment across available pairs."""
    pairs, _ = available_variant_pairs(hours, variants)
    rows = []
    if pairs:
        daily = daily_mae(hours, sorted({name for pair in pairs for name in pair}))
        for variant, base in pairs:
            mae_base = (hours[base] - hours["actual"]).abs().mean()
            mae_variant = (hours[variant] - hours["actual"]).abs().mean()
            p_raw = diebold_mariano(daily, variant, base, lags=lags)[1]
            rows.append({
                "variant": variant,
                "base": base,
                "MAE base": mae_base,
                "MAE variant": mae_variant,
                "difference": mae_variant - mae_base,
                "difference %": 100 * (mae_variant - mae_base) / mae_base,
                "mean daily MAE difference": (daily[variant] - daily[base]).mean(),
                "raw p-value": p_raw,
            })
    columns = ["variant", "base", "MAE base", "MAE variant", "difference", "difference %",
               "mean daily MAE difference", "raw p-value"]
    table = pd.DataFrame(rows, columns=columns).set_index("variant")
    if not table.empty:
        adjusted = adjust_test_family(table["raw p-value"].to_numpy())
        table["Holm-adjusted p-value"] = adjusted
        table["significant after adjustment"] = adjusted <= ALPHA
    return table


def to_markdown(table: pd.DataFrame, include_index: bool = True) -> str:
    """A plain markdown table, without extra dependencies."""
    def cell(value) -> str:
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    header = ([table.index.name or ""] if include_index else []) + list(table.columns)
    lines = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    # itertuples keeps each column's type, so hour counts stay integers (iterrows would turn them into floats).
    rows = table.itertuples(name=None) if include_index else table.itertuples(index=False, name=None)
    for row in rows:
        values = [str(row[0])] if include_index else []
        values.extend(cell(value) for value in (row[1:] if include_index else row))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def mae_chart(hours: pd.DataFrame, models: list[str]) -> Figure:
    """The README's headline chart: one bar per main model, best first, with its papers' setup, if run, just below."""
    shown = models + [PAPERS_SETUP[model] for model in models if PAPERS_SETUP.get(model) in hours]
    mae = hours[shown].sub(hours["actual"], axis=0).abs().mean()
    bars = []  # (position, label, model), top to bottom
    position = 0.0
    for model in sorted(models, key=mae.get):
        bars.append((position, MODEL_LABELS.get(model, model), model))
        if PAPERS_SETUP.get(model) in mae:
            position += 0.4
            bars.append((position, "papers' setup", PAPERS_SETUP[model]))
        position += 0.7
    positions, labels, names = zip(*bars)

    fig = Figure(figsize=(8, 3.6), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.barh(positions, mae[list(names)], height=0.34, color=[MODEL_COLOURS.get(name, BASELINE) for name in names])
    for position, name in zip(positions, names):  # each value at the end of its bar, in text colour, as in results.md
        ax.text(mae[name] + 0.4, position, f"{mae[name]:.2f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()  # best model at the top, as in the README's table
    ax.set_xlim(0, 1.12 * mae.max())
    ax.set_xlabel("MAE, EUR/MWh", color=INK_2)
    ax.tick_params(color=MUTED, labelcolor=INK_2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set(color=BASELINE, linewidth=0.8)
    first, last = (timestamp.tz_convert(LOCAL_TZ) for timestamp in (hours.index.min(), hours.index.max()))
    ax.set_title(f"Mean absolute error by model, {day_label(first, weekday=False)} to {day_label(last, weekday=False)}",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    return fig


def main() -> None:
    con = store.connect(read_only=True)
    predictions = con.sql("SELECT model, ts_utc, prediction, actual FROM predictions").df()
    con.close()
    unregistered = sorted(set(predictions["model"]) - set(MODELS))
    if unregistered:
        print(f"ignored: rows of models no longer registered ({', '.join(unregistered)})\n")
    predictions = predictions[predictions["model"].isin(MODELS)]

    models = sorted(predictions["model"].unique(), key=lambda model: (model != BENCHMARK, model))
    hours = common_hours(predictions)  # every model, variants included, is scored on the same hours
    if hours.empty:
        raise ValueError("no common hours across all stored models; check missing forecasts and model coverage")
    main_models = [model for model in models if model not in VARIANTS]
    if not main_models:
        raise ValueError("no main-model predictions available")
    missing_main = sorted(set(MODELS) - set(VARIANTS) - set(main_models))
    coverage_note = (f" Missing registered main models: {', '.join(missing_main)}. This is a partial report."
                     if missing_main else "")
    excluded_hours = predictions["ts_utc"].nunique() - len(hours)
    print(f"Common-sample restriction excludes {excluded_hours:,} stored hours.{coverage_note}")
    first, last = hours.index.min(), hours.index.max()
    print(f"{len(hours):,} hours forecast by every model ({', '.join(models)}), "
          f"{first.tz_convert(LOCAL_TZ):%Y-%m-%d} to {last.tz_convert(LOCAL_TZ):%Y-%m-%d}\n")

    sections = {
        "Overall": scores(hours, main_models),
        "By period": scores(hours, main_models, "period"),
        "By year": scores(hours, main_models, "year"),
        "By month": scores(hours, main_models, "month"),
        "By local hour": scores(hours, main_models, "local_hour"),
    }
    overall = sections["Overall"]
    if SIMILAR_DAY in main_models:
        for model in main_models:
            if model != SIMILAR_DAY:
                overall[f"% vs {SIMILAR_DAY} {model}"] = 100 * (overall[f"MAE {model}"] / overall[f"MAE {SIMILAR_DAY}"] - 1)
    for title in ["Overall", "By period", "By year"]:
        print(title)
        print(sections[title].round(2).to_string(), "\n")

    ranked = sorted(main_models, key=lambda model: overall.loc["all hours", f"MAE {model}"])  # best first
    daily_main = daily_mae(hours, ranked)
    day_count = len(daily_main)
    main_tests = main_comparison_table(daily_main, ranked)
    print(f"Main-model two-sided Diebold-Mariano tests ({day_count:,} days; Holm-adjusted across {len(main_tests)} pairs)")
    print(main_tests.to_string(index=False), "\n")

    available_variants, missing_variants = available_variant_pairs(hours, VARIANTS)
    variants = variant_table(hours, VARIANTS)
    if not variants.empty:
        print(f"Exploratory variant tests against their base models (Holm-adjusted across {len(variants)} pairs)")
        print(variants.round(2).to_string(), "\n")
    if missing_variants:
        missing_text = ", ".join(f"{variant} vs {base}" for variant, base in missing_variants)
        print(f"warning: missing registered variant comparisons ({missing_text})\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Backtest results\n\nMAE in EUR/MWh on {len(hours):,} hours forecast by every stored registered model, "
            f"{day_label(first.tz_convert(LOCAL_TZ), weekday=False)} to {day_label(last.tz_convert(LOCAL_TZ), weekday=False)}. "
            f"The common-sample restriction excludes {excluded_hours:,} stored hours.{coverage_note}"]
    body += [f"## {title}\n\n{to_markdown(table)}" for title, table in sections.items()]
    main_report = format_report_p_values(main_tests, ["Raw two-sided p-value", "Holm-adjusted p-value"])
    main_report["Significant after adjustment"] = main_report["Significant after adjustment"].map({True: "Yes", False: "No"})
    main_report.loc[main_tests["Raw two-sided p-value"].isna(), "Significant after adjustment"] = "Not testable"
    body += [f"## Main-model Diebold-Mariano comparisons\n\nTwo-sided tests of equal predictive accuracy on daily MAE over {day_count:,} days. "
             f"Difference is Model B minus Model A in EUR/MWh; positive values favour Model A. "
             f"The raw p-values use a Newey-West HAC standard error with {DM_LAGS} Bartlett lags; Holm adjustment "
             f"covers the {len(main_tests)} available unique pairs. Significance threshold: {ALPHA:.0%}. P-values are shown at full precision. "
             "Comparisons with fewer than two days or non-positive estimated variance are marked not testable. "
             "They remain in the adjustment family with p=1 internally; no test result is claimed for them.\n\n"
             + (to_markdown(main_report, include_index=False) if len(main_tests) else "At least two main models are needed for a comparison.")]
    variant_intro = ("## Exploratory variant comparisons\n\nEach variant changes one input or setting in its corresponding main model. "
                     "The mean daily MAE difference is variant minus main model; negative values favour the variant. "
                     "Two-sided DM p-values use the seven-lag HAC standard error. Holm adjustment is a separate family "
                     f"across {len(available_variants)} of {len(VARIANTS)} registered comparisons available in this report; "
                     "significance threshold: 5%. "
                     "These are secondary sensitivities. Multiplicity adjustment does not remove the limitations of selecting "
                     "or evaluating variants on the same test period. Untestable comparisons follow the same convention as the main-model family.")
    if missing_variants:
        missing_text = ", ".join(f"`{variant}` versus `{base}`" for variant, base in missing_variants)
        variant_intro += (f" Missing registered comparisons: {missing_text}. The Holm adjustment covers only the "
                          f"{len(available_variants)} available comparisons.")
    if not variants.empty:
        variant_report = variants[["base", "mean daily MAE difference", "raw p-value",
                                   "Holm-adjusted p-value", "significant after adjustment"]].copy()
        variant_report = variant_report.rename(columns={
            "base": "Corresponding main model",
            "mean daily MAE difference": "Mean daily MAE difference (variant − main)",
            "raw p-value": "Raw two-sided p-value",
            "Holm-adjusted p-value": "Holm-adjusted p-value",
            "significant after adjustment": "Significant after adjustment",
        })
        variant_report["Significant after adjustment"] = variant_report["Significant after adjustment"].map(
            {True: "Yes", False: "No"})
        variant_report.loc[variants["raw p-value"].isna(), "Significant after adjustment"] = "Not testable"
        variant_report = format_report_p_values(variant_report, ["Raw two-sided p-value", "Holm-adjusted p-value"])
        body += [variant_intro + "\n\n" + to_markdown(variant_report)]
    else:
        body += [variant_intro + " No registered variant/base pair is available for comparison."]
    acf_table = autocorrelation_diagnostic(daily_main, ranked)
    bandwidth_table = bandwidth_sensitivity_table(daily_main, ranked)
    acf_report = acf_table.copy()
    acf_report["Largest absolute ACF, lags 8–28"] = acf_report["Largest absolute ACF, lags 8–28"].map(
        lambda value: f"{value:.3f}")
    body += [f"## HAC bandwidth robustness diagnostic\n\nSample autocorrelations of the {len(main_tests)} available main-model daily loss-difference series at lags 8–28 range from "
             f"{acf_table['Largest absolute ACF, lags 8–28'].abs().min():.3f} to "
             f"{acf_table['Largest absolute ACF, lags 8–28'].abs().max():.3f} for the pair-specific largest "
             "absolute coefficient shown below (signed values). The table reports two-sided raw p-values with Bartlett "
             "Newey-West bandwidths of 7, 14 and 28 days. This is a robustness diagnostic, not an additional "
             "testing family; the seven-lag specification remains primary. The table permits comparison of the "
             "raw p-values across bandwidths.\n\n"
             + to_markdown(acf_report, include_index=False) + "\n\n"
             + to_markdown(format_report_p_values(bandwidth_table, list(bandwidth_table.columns[2:])),
                           include_index=False)]
    if acf_table.empty or acf_table["Largest absolute ACF, lags 8–28"].isna().all():
        body[-1] = ("## HAC bandwidth robustness diagnostic\n\nNo estimable autocorrelation diagnostic: "
                    "at least two models, nine days and a non-constant loss difference are needed.")
    RESULTS.write_text("\n\n".join(body) + "\n")
    print(f"all tables written to {RESULTS.relative_to(PROJECT_ROOT)}")

    save(mae_chart(hours, main_models), CHART, SOURCE_BACKTEST)
    print(f"headline chart written to {(FIG_DIR / CHART).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
