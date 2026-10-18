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
1 October 2025), by year, by month and by local hour, and written to reports/results.md.

Only models registered in models/__init__.py are scored. Rows left in the predictions table
by a model that no longer exists (an old name, say) are listed and ignored.

Main against comparison version. The main linear and tree models use the renewables forecast
for day d, as the papers do; it is published at 18:00 on d-1, after the issue time. Their
comparison versions use the forecast for d-1 instead. A separate table shows, for each pair,
how much lower the main version's error is and whether that gap is significant: what the
papers' convention is worth.

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

from pl_dayahead import store
from pl_dayahead.config import PROJECT_ROOT, SDAC_15MIN_START
from pl_dayahead.models import MODELS
from pl_dayahead.timeutils import LOCAL_TZ

BENCHMARK = "naive_weekly"
VERSION_PAIRS = {"linear": "linear_res_d1", "gbm": "gbm_res_d1"}  # main -> comparison
RESULTS = PROJECT_ROOT / "reports" / "results.md"
PERIODS = ["hourly MTU (to Sep 2025)", "15-minute MTU (from Oct 2025)"]
DM_LAGS = 7  # days of autocorrelation allowed for in the Diebold-Mariano standard error


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


def version_gaps(wide: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """For each main model whose comparison version was also run: both MAEs, the gap and its p-value.

    The gap is the comparison version's MAE minus the main version's: how much the renewables
    forecast for day d (published after the issue time) lowers the error. The p-value is the
    one-sided Diebold-Mariano test that the main version is more accurate.
    """
    pairs = [(m, c) for m, c in VERSION_PAIRS.items() if m in models and c in models]
    rows = []
    if pairs:
        daily = daily_mae(wide, [name for pair in pairs for name in pair])
        for main, comparison in pairs:
            mae_main = (wide[main] - wide["actual"]).abs().mean()
            mae_comparison = (wide[comparison] - wide["actual"]).abs().mean()
            rows.append({
                "model": main,
                "MAE main": mae_main,
                "MAE comparison": mae_comparison,
                "gap": mae_comparison - mae_main,
                "gap %": 100 * (mae_comparison - mae_main) / mae_comparison,
                "p main better": fmt_p(diebold_mariano(daily, main, comparison)[1]),
            })
    table = pd.DataFrame(rows, columns=["model", "MAE main", "MAE comparison", "gap", "gap %", "p main better"])
    return table.set_index("model")


def to_markdown(df: pd.DataFrame) -> str:
    """A plain markdown table, without extra dependencies."""
    cols = [df.index.name or ""] + list(df.columns)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    # itertuples keeps each column's type, so hour counts stay integers (iterrows would turn them into floats)
    for idx, *values in df.itertuples(name=None):
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in values]) + " |")
    return "\n".join(lines)


def main() -> None:
    con = store.connect()
    predictions = con.sql("SELECT model, ts_utc, prediction, actual FROM predictions").df()
    con.close()
    unregistered = sorted(set(predictions["model"]) - set(MODELS))
    if unregistered:
        print(f"ignored: rows of models no longer registered ({', '.join(unregistered)})\n")
    predictions = predictions[predictions["model"].isin(MODELS)]

    models = sorted(predictions["model"].unique(), key=lambda m: (m != BENCHMARK, m))
    wide = common_hours(predictions)
    first, last = wide.index.min(), wide.index.max()
    print(f"{len(wide):,} hours forecast by every model ({', '.join(models)}), "
          f"{first.tz_convert(LOCAL_TZ):%Y-%m-%d} to {last.tz_convert(LOCAL_TZ):%Y-%m-%d}\n")

    sections = {
        "Overall": scores(wide, models),
        "By period": scores(wide, models, "period"),
        "By year": scores(wide, models, "year"),
        "By month": scores(wide, models, "month"),
        "By local hour": scores(wide, models, "local_hour"),
    }
    for title in ["Overall", "By period", "By year"]:
        print(title)
        print(sections[title].round(2).to_string(), "\n")

    ranked = sorted(models, key=lambda m: sections["Overall"].loc["all hours", f"MAE {m}"])  # best first
    dm = format_p(dm_table(wide, ranked))
    n_days = len(daily_mae(wide, ranked))
    print(f"Diebold-Mariano p-values on daily MAE ({n_days:,} days, best model first). "
          "Below 0.05: the row model is more accurate than the column model.")
    print(dm.to_string(), "\n")

    gaps = version_gaps(wide, models)
    if not gaps.empty:
        print("Main against comparison version: what the day-d renewables forecast is worth "
              "(gap = comparison MAE minus main MAE, EUR/MWh)")
        print(gaps.round(2).to_string(), "\n")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Backtest results\n\nMAE in EUR/MWh on {len(wide):,} hours forecast by every model."]
    body += [f"## {t}\n\n{to_markdown(df)}" for t, df in sections.items()]
    body += [f"## Diebold-Mariano test\n\nOne-sided p-values on daily MAE over {n_days:,} days, "
             f"Newey-West standard error with {DM_LAGS} lags. Below 0.05: the row model is more "
             f"accurate than the column model.\n\n{to_markdown(dm)}"]
    if not gaps.empty:
        body += ["## Main against comparison version\n\nThe main models use the renewables forecast for "
                 "day d, published at 18:00 on d-1, as the papers do; the comparison versions use the "
                 "forecast for d-1, public at 11:00. Gap = comparison MAE minus main MAE in EUR/MWh; "
                 f"p = one-sided Diebold-Mariano test that the main version is more accurate.\n\n{to_markdown(gaps)}"]
    RESULTS.write_text("\n\n".join(body) + "\n")
    print(f"all tables written to {RESULTS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
