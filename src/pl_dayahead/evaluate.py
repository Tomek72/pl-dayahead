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
from pl_dayahead.config import PROJECT_ROOT, SDAC_15MIN_START
from pl_dayahead.models import MODELS, VARIANTS
from pl_dayahead.timeutils import LOCAL_TZ

BENCHMARK = "naive_weekly"
SIMILAR_DAY = "naive_similar_day"  # the stronger benchmark, also reported as a percentage change
RESULTS = PROJECT_ROOT / "reports" / "results.md"
FIGURE = PROJECT_ROOT / "reports" / "figures" / "mae_by_model.png"
PERIODS = ["hourly MTU (to Sep 2025)", "15-minute MTU (from Oct 2025)"]
DM_LAGS = 7  # days of autocorrelation allowed for in the Diebold-Mariano standard error
PAPERS_SETUP = {"gbm": "gbm_papers", "linear": "linear_papers"}  # main model -> its variant with day d's renewables

# The headline chart, in the notebooks' style. Each model keeps its colour (gbm blue, the linear model orange), its
# papers' setup takes a lighter step of the same hue, and the naive benchmarks are grey.
LABELS = {"gbm": "Gradient-boosted trees", "linear": "Linear expert model",
          "naive_similar_day": "Similar-day naive", "naive_weekly": "Weekly naive"}
COLOURS = {"gbm": "#2a78d6", "gbm_papers": "#86b6ef", "linear": "#eb6834", "linear_papers": "#eb9c7f"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"  # text: primary, secondary, muted
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"  # gridlines, axis line and benchmark bars, background
SOURCE = "Source: ENTSO-E Transparency Platform; own backtest"


def common_hours(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per hour, one prediction column per model; only hours every model forecast."""
    wide = predictions.pivot_table(index="ts_utc", columns="model", values="prediction", aggfunc="first")
    actual = predictions.groupby("ts_utc")["actual"].first()
    wide = wide.join(actual).dropna()
    local = wide.index.tz_convert(LOCAL_TZ)
    after = local.date >= SDAC_15MIN_START
    # An ordered category, so tables list the older period first.
    wide["period"] = pd.Categorical.from_codes(after.astype(int), categories=PERIODS, ordered=True)
    wide["year"] = local.year
    wide["month"] = local.strftime("%Y-%m")
    wide["local_hour"] = local.hour
    return wide


def scores(wide: pd.DataFrame, models: list[str], by: str | None = None) -> pd.DataFrame:
    """MAE per model, and relative MAE against the benchmark, overall or per group."""
    errors = wide[models].sub(wide["actual"], axis=0).abs()
    if by is not None:
        errors[by] = wide[by]
        groups = errors.groupby(by, observed=True)  # observed: skip a period with no hours yet
        mae = groups[models].mean()
        hours = groups.size()
    else:
        mae = errors[models].mean().to_frame("all hours").T
        hours = pd.Series([len(errors)], index=mae.index)
    table = mae.add_prefix("MAE ")
    if BENCHMARK in models:
        for m in models:
            if m != BENCHMARK:
                table[f"rel MAE {m}"] = mae[m] / mae[BENCHMARK]
    table["hours"] = hours
    return table


def daily_mae(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per local delivery day, each model's MAE over that day's hours."""
    day = wide.index.tz_convert(LOCAL_TZ).date
    return wide[models].sub(wide["actual"], axis=0).abs().groupby(day).mean()


def diebold_mariano(daily: pd.DataFrame, better: str, worse: str, lags: int = DM_LAGS) -> tuple[float, float]:
    """Test whether `better` has lower daily MAE than `worse`. Returns (statistic, one-sided p-value)."""
    d = (daily[worse] - daily[better]).to_numpy()  # positive on days `better` wins
    n = len(d)
    u = d - d.mean()
    var = u @ u / n
    for k in range(1, min(lags, n - 1) + 1):  # Newey-West: add the covariance with k days earlier
        var += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    if var <= 0:
        return math.nan, math.nan  # identical forecasts: nothing to test
    stat = d.mean() / math.sqrt(var / n)
    p = 0.5 * math.erfc(stat / math.sqrt(2))  # P(standard normal > stat)
    return stat, p


def dm_table(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """p-value that the row model is more accurate than the column model, for every pair."""
    daily = daily_mae(wide, models)
    table = pd.DataFrame(np.nan, index=models, columns=models)
    for row in models:
        for col in models:
            if row != col:
                table.loc[row, col] = diebold_mariano(daily, row, col)[1]
    table.index.name = "row better than column?"
    return table


def fmt_p(p: float) -> str:
    """A p-value as text: '<0.001' for tiny ones, blank if there was nothing to test."""
    if math.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def format_p(table: pd.DataFrame) -> pd.DataFrame:
    """A table of p-values as text, blank on the diagonal."""
    return table.apply(lambda col: col.map(fmt_p))


def variant_table(wide: pd.DataFrame, variants: dict[str, str]) -> pd.DataFrame:
    """For each variant run together with its base model: both MAEs, the difference and its p-value.

    The difference is the variant's MAE minus the base model's, so a negative number means the
    variant is more accurate. The p-value is two-sided: twice the smaller of the two one-sided
    Diebold-Mariano p-values.
    """
    pairs = [(v, b) for v, b in variants.items() if v in wide and b in wide]
    rows = []
    if pairs:
        daily = daily_mae(wide, sorted({name for pair in pairs for name in pair}))
        for variant, base in pairs:
            mae_base = (wide[base] - wide["actual"]).abs().mean()
            mae_variant = (wide[variant] - wide["actual"]).abs().mean()
            p = diebold_mariano(daily, variant, base)[1]
            rows.append({
                "variant": variant,
                "base": base,
                "MAE base": mae_base,
                "MAE variant": mae_variant,
                "difference": mae_variant - mae_base,
                "difference %": 100 * (mae_variant - mae_base) / mae_base,
                "p": fmt_p(2 * min(p, 1 - p)),
            })
    columns = ["variant", "base", "MAE base", "MAE variant", "difference", "difference %", "p"]
    return pd.DataFrame(rows, columns=columns).set_index("variant")


def to_markdown(df: pd.DataFrame) -> str:
    """A plain markdown table, without extra dependencies."""
    cols = [df.index.name or ""] + list(df.columns)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    # itertuples keeps each column's type, so hour counts stay integers (iterrows would turn them into floats)
    for idx, *values in df.itertuples(name=None):
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in values]) + " |")
    return "\n".join(lines)


def mae_chart(wide: pd.DataFrame, models: list[str]) -> Figure:
    """The README's headline chart: one bar per main model, best first, with its papers' setup, if run, just below."""
    shown = models + [PAPERS_SETUP[m] for m in models if PAPERS_SETUP.get(m) in wide]
    mae = wide[shown].sub(wide["actual"], axis=0).abs().mean()
    bars = []  # (position, label, model), top to bottom
    y = 0.0
    for model in sorted(models, key=mae.get):
        bars.append((y, LABELS.get(model, model), model))
        if PAPERS_SETUP.get(model) in mae:
            y += 0.4
            bars.append((y, "papers' setup", PAPERS_SETUP[model]))
        y += 0.7
    positions, labels, names = zip(*bars)

    fig = Figure(figsize=(8, 3.6), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.barh(positions, mae[list(names)], height=0.34, color=[COLOURS.get(n, BASELINE) for n in names])
    for y, name in zip(positions, names):  # each value at the end of its bar, in text colour, as in results.md
        ax.text(mae[name] + 0.4, y, f"{mae[name]:.2f}", va="center", fontsize=9, color=INK_2)
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
    first, last = (t.tz_convert(LOCAL_TZ) for t in (wide.index.min(), wide.index.max()))
    ax.set_title(f"Mean absolute error by model, {first.day} {first:%b %Y} to {last.day} {last:%b %Y}",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    fig.text(0.01, 0.01, SOURCE, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def main() -> None:
    con = store.connect()
    predictions = con.sql("SELECT model, ts_utc, prediction, actual FROM predictions").df()
    con.close()
    unregistered = sorted(set(predictions["model"]) - set(MODELS))
    if unregistered:
        print(f"ignored: rows of models no longer registered ({', '.join(unregistered)})\n")
    predictions = predictions[predictions["model"].isin(MODELS)]

    models = sorted(predictions["model"].unique(), key=lambda m: (m != BENCHMARK, m))
    wide = common_hours(predictions)  # every model, variants included, is scored on the same hours
    main_models = [m for m in models if m not in VARIANTS]
    first, last = wide.index.min(), wide.index.max()
    print(f"{len(wide):,} hours forecast by every model ({', '.join(models)}), "
          f"{first.tz_convert(LOCAL_TZ):%Y-%m-%d} to {last.tz_convert(LOCAL_TZ):%Y-%m-%d}\n")

    sections = {
        "Overall": scores(wide, main_models),
        "By period": scores(wide, main_models, "period"),
        "By year": scores(wide, main_models, "year"),
        "By month": scores(wide, main_models, "month"),
        "By local hour": scores(wide, main_models, "local_hour"),
    }
    overall = sections["Overall"]
    if SIMILAR_DAY in main_models:
        for m in main_models:
            if m != SIMILAR_DAY:
                overall[f"% vs {SIMILAR_DAY} {m}"] = 100 * (overall[f"MAE {m}"] / overall[f"MAE {SIMILAR_DAY}"] - 1)
    for title in ["Overall", "By period", "By year"]:
        print(title)
        print(sections[title].round(2).to_string(), "\n")

    ranked = sorted(main_models, key=lambda m: overall.loc["all hours", f"MAE {m}"])  # best first
    dm = format_p(dm_table(wide, ranked))
    n_days = len(daily_mae(wide, ranked))
    print(f"Diebold-Mariano p-values on daily MAE ({n_days:,} days, best model first). "
          "Below 0.05: the row model is more accurate than the column model.")
    print(dm.to_string(), "\n")

    variants = variant_table(wide, VARIANTS)
    if not variants.empty:
        print("Variants against their base models (difference = variant MAE minus base MAE, EUR/MWh; "
              "p two-sided)")
        print(variants.round(2).to_string(), "\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Backtest results\n\nMAE in EUR/MWh on {len(wide):,} hours forecast by every model."]
    body += [f"## {t}\n\n{to_markdown(df)}" for t, df in sections.items()]
    body += [f"## Diebold-Mariano test\n\nOne-sided p-values on daily MAE over {n_days:,} days, "
             f"Newey-West standard error with {DM_LAGS} lags. Below 0.05: the row model is more "
             f"accurate than the column model.\n\n{to_markdown(dm)}"]
    if not variants.empty:
        body += ["## Variants against their base models\n\nEach variant changes one thing about a main "
                 "model (see models/linear.py and models/boosting.py). Difference = variant MAE minus base "
                 "MAE in EUR/MWh; p = two-sided Diebold-Mariano test on daily MAE.\n\n" + to_markdown(variants)]
    RESULTS.write_text("\n\n".join(body) + "\n")
    print(f"all tables written to {RESULTS.relative_to(PROJECT_ROOT)}")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    mae_chart(wide, main_models).savefig(FIGURE, dpi=150)
    print(f"headline chart written to {FIGURE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
