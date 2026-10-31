"""Compare models on exactly the same hours.

Run from the project root, after one or more backtests:

    uv run python -m pl_dayahead.evaluate

Every model is scored only on hours where all models made a forecast and the actual price
is known, so no model looks better by skipping hard hours.

Metrics:

* MAE, mean absolute error in EUR/MWh: the average size of the miss. It is used rather than
  squared error so that a handful of extreme hours (spikes, deep negative prices) cannot
  dominate the comparison.
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
papers' setup, which uses the renewables forecast for day d, published at 18:00 on d-1, after the
issue time; two robustness checks; and the carbon variants, which add the price of the latest EU
emission allowance auction public at 11:00. They are scored on the same hours as every other model
but kept out of the main tables. A separate table compares each with its base model: the
difference in MAE, and a two-sided Diebold-Mariano p-value, the chance of a difference this
large if the two were equally accurate.

Is a difference real or luck? A lower MAE over 24,000 hours can still come from a handful of
lucky weeks. The Diebold-Mariano test, the standard check in the price forecasting literature,
asks whether one model's daily error is lower than another's on average by more than chance
would produce:

1. For each delivery day, take each model's MAE over that day's hours. Hours within one day
   share the same weather and fuel news, so their errors are not independent; days are the
   unit of evidence (about 1,000 of them), not hours.
2. Take the daily difference between the two models and test whether its average is above
   zero, like a t-test on the differences.
3. Consecutive days are also alike (a volatile week is volatile all week), so the standard
   error uses a Newey-West correction over 7 days of lags. That makes the test stricter than
   the plain version described in Lago et al. (2021).

The daily mean absolute error is Lago et al.'s daily loss (the sum of absolute errors over the
day's hours, the L1 norm) divided by the number of hours. Dividing every day by 24 leaves the
test statistic unchanged; only the 23- and 25-hour clock-change days are scaled slightly
differently.

The p-value is the chance of seeing an advantage this large if the row model were in fact no
better than the column model. Below 0.05 is the usual bar for "better".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pl_dayahead import store
from pl_dayahead.config import FIG_DIR, PROJECT_ROOT, SDAC_15MIN_START
from pl_dayahead.models import MODELS, VARIANTS
from pl_dayahead.style import BASELINE, GRID, INK, INK_2, MODEL_COLOURS, MUTED, SOURCE_BACKTEST, SURFACE, save
from pl_dayahead.timeutils import LOCAL_TZ, day_label

BENCHMARK = "naive_weekly"
SIMILAR_DAY = "naive_similar_day"  # the stronger benchmark, also reported as a percentage change
RESULTS = PROJECT_ROOT / "reports" / "results.md"
CHART = "mae_by_model.png"
PERIODS = ["hourly MTU (to Sep 2025)", "15-minute MTU (from Oct 2025)"]
DM_LAGS = 7  # days of autocorrelation allowed for in the Diebold-Mariano standard error
PAPERS_SETUP = {"gbm": "gbm_papers", "linear": "linear_papers"}  # main model -> its variant with day d's renewables
MODEL_LABELS = {"gbm": "Gradient-boosted trees", "linear": "Linear expert model",
                "naive_similar_day": "Similar-day naive", "naive_weekly": "Weekly naive"}


def common_hours(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per hour, one prediction column per model; only hours every model forecast."""
    hours = predictions.pivot_table(index="ts_utc", columns="model", values="prediction", aggfunc="first")
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


def diebold_mariano(daily: pd.DataFrame, better: str, worse: str, lags: int = DM_LAGS) -> tuple[float, float]:
    """Test whether `better` has lower daily MAE than `worse`. Returns (statistic, one-sided p-value)."""
    difference = (daily[worse] - daily[better]).to_numpy()  # positive on days `better` wins
    days = len(difference)
    centred = difference - difference.mean()
    variance = centred @ centred / days
    for lag in range(1, min(lags, days - 1) + 1):  # Newey-West: add the covariance with `lag` days earlier
        variance += 2 * (1 - lag / (lags + 1)) * (centred[lag:] @ centred[:-lag]) / days
    if variance <= 0:
        return math.nan, math.nan  # identical forecasts: nothing to test
    statistic = difference.mean() / math.sqrt(variance / days)
    return statistic, 0.5 * math.erfc(statistic / math.sqrt(2))  # P(standard normal > statistic)


def dm_table(hours: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """p-value that the row model is more accurate than the column model, for every pair."""
    daily = daily_mae(hours, models)
    table = pd.DataFrame(np.nan, index=models, columns=models)
    for row in models:
        for column in models:
            if row != column:
                table.loc[row, column] = diebold_mariano(daily, row, column)[1]
    table.index.name = "row better than column?"
    return table


def format_p(p_value: float) -> str:
    """A p-value as text: '<0.001' for tiny ones, blank if there was nothing to test."""
    if math.isnan(p_value):
        return ""
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def variant_table(hours: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """For each variant run together with its base model: both MAEs, the difference and its p-value.

    The difference is the variant's MAE minus the base model's, so a negative number means the
    variant is more accurate. The p-value is two-sided: twice the smaller of the two one-sided
    Diebold-Mariano p-values.
    """
    pairs = [(variant, base) for variant, base in variants.items() if variant in hours and base in hours]
    rows = []
    if pairs:
        daily = daily_mae(hours, sorted({name for pair in pairs for name in pair}))
        for variant, base in pairs:
            mae_base = (hours[base] - hours["actual"]).abs().mean()
            mae_variant = (hours[variant] - hours["actual"]).abs().mean()
            p_value = diebold_mariano(daily, variant, base)[1]
            rows.append({
                "variant": variant,
                "base": base,
                "MAE base": mae_base,
                "MAE variant": mae_variant,
                "difference": mae_variant - mae_base,
                "difference %": 100 * (mae_variant - mae_base) / mae_base,
                "p": format_p(2 * min(p_value, 1 - p_value)),
            })
    columns = ["variant", "base", "MAE base", "MAE variant", "difference", "difference %", "p"]
    return pd.DataFrame(rows, columns=columns).set_index("variant")


def to_markdown(table: pd.DataFrame) -> str:
    """A plain markdown table, without extra dependencies."""
    def cell(value) -> str:
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    header = [table.index.name or ""] + list(table.columns)
    lines = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    # itertuples keeps each column's type, so hour counts stay integers (iterrows would turn them into floats)
    for label, *values in table.itertuples(name=None):
        lines.append("| " + " | ".join([str(label)] + [cell(value) for value in values]) + " |")
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
    con = store.connect()
    predictions = con.sql("SELECT model, ts_utc, prediction, actual FROM predictions").df()
    con.close()
    unregistered = sorted(set(predictions["model"]) - set(MODELS))
    if unregistered:
        print(f"ignored: rows of models no longer registered ({', '.join(unregistered)})\n")
    predictions = predictions[predictions["model"].isin(MODELS)]

    models = sorted(predictions["model"].unique(), key=lambda model: (model != BENCHMARK, model))
    hours = common_hours(predictions)  # every model, variants included, is scored on the same hours
    main_models = [model for model in models if model not in VARIANTS]
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
    dm_p_values = dm_table(hours, ranked).map(format_p)
    day_count = len(daily_mae(hours, ranked))
    print(f"Diebold-Mariano p-values on daily MAE ({day_count:,} days, best model first). "
          "Below 0.05: the row model is more accurate than the column model.")
    print(dm_p_values.to_string(), "\n")

    variants = variant_table(hours, VARIANTS)
    if not variants.empty:
        print("Variants against their base models (difference = variant MAE minus base MAE, EUR/MWh; "
              "p two-sided)")
        print(variants.round(2).to_string(), "\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Backtest results\n\nMAE in EUR/MWh on {len(hours):,} hours forecast by every model."]
    body += [f"## {title}\n\n{to_markdown(table)}" for title, table in sections.items()]
    body += [f"## Diebold-Mariano test\n\nOne-sided p-values on daily MAE over {day_count:,} days, "
             f"Newey-West standard error with {DM_LAGS} lags. Below 0.05: the row model is more "
             f"accurate than the column model.\n\n{to_markdown(dm_p_values)}"]
    if not variants.empty:
        body += ["## Variants against their base models\n\nEach variant changes one thing about a main "
                 "model (see models/linear.py and models/boosting.py). Difference = variant MAE minus base "
                 "MAE in EUR/MWh; p = two-sided Diebold-Mariano test on daily MAE.\n\n" + to_markdown(variants)]
    RESULTS.write_text("\n\n".join(body) + "\n")
    print(f"all tables written to {RESULTS.relative_to(PROJECT_ROOT)}")

    save(mae_chart(hours, main_models), CHART, SOURCE_BACKTEST)
    print(f"headline chart written to {(FIG_DIR / CHART).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
