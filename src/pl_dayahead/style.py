"""The look every chart and table shares, in evaluate.py and both notebooks: colours, settings and saving.

The colours come from a palette tested for colour-blind readability. Each model keeps one colour in
every chart, and its papers' setup takes a lighter step of the same hue.
"""

from __future__ import annotations

import matplotlib

from pl_dayahead.config import FIG_DIR

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"  # text: primary, secondary, muted
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"  # gridlines, axis line and benchmark bars, background
BLUE = "#2a78d6"
MODEL_COLOURS = {"actual": INK, "gbm": BLUE, "gbm_papers": "#86b6ef", "linear": "#eb6834", "linear_papers": "#eb9c7f"}
SOURCE_ENTSOE = "Source: ENTSO-E Transparency Platform"
SOURCE_BACKTEST = SOURCE_ENTSOE + "; own backtest"

NOTEBOOK_SETTINGS = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,  # light horizontal gridlines behind the data
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
}


def use_notebook_style() -> None:
    """Apply the notebooks' chart settings. evaluate.py styles its one chart itself."""
    matplotlib.rcParams.update(NOTEBOOK_SETTINGS)


def save(fig, name: str, source: str, rect: tuple = (0, 0.04, 1, 1)) -> None:
    """Put the source line under the chart, fit the layout into `rect` and save it to reports/figures as `name`."""
    fig.text(0.01, 0.01, source, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=rect)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150)


def whole(value: float) -> str:
    """A value rounded to a whole number for a table, without the '-0' that rounding a small negative leaves."""
    text = f"{value:.0f}"
    return "0" if text == "-0" else text
