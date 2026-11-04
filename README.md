# Polish day-ahead electricity price forecast

This project forecasts the 24 hourly prices of the Polish day-ahead auction at 11:00 Warsaw time on the day before delivery, one hour before bids close, using only information published by then. A linear expert model and gradient-boosted trees are tested against two naive benchmarks, which copy earlier prices, in a walk-forward backtest over 1,028 delivery days, from 1 January 2024 to 24 October 2026.

Both models cut the mean absolute error (MAE) by 22% against the similar-day naive benchmark, to 21.18 EUR/MWh for the trees and 21.30 for the linear model, against 27.34; the difference between the two models is not significant. The trees' daily bias, their mean forecast error over the day, follows the change from the wind and solar forecast they saw at 11:00, the one for the day before delivery, to the delivery day's own, published at 18:00 (Spearman rank correlation 0.59 over all days). Given that day's forecast, the same models' error would fall by 14 to 20%, which suggests that a better wind and solar forecast at 11:00 would help a desk more than a more complex model.

![Mean absolute error by model](reports/figures/mae_by_model.png)

*Mean absolute error on the 24,671 hours every model forecast. The lighter bars are the papers' setup: the same models given the delivery day's wind and solar forecast, which is due only after bids close. From `src/pl_dayahead/evaluate.py`.*

## The forecasting problem

Poland's day-ahead electricity price is set in Single Day-Ahead Coupling, the European day-ahead auction. Bids for delivery day d close at 12:00 Warsaw time on d−1, the gate closure, and the auction sets a price for every hour of d in EUR/MWh, or every quarter-hour since 1 October 2025. The target is the hourly price, since then the mean of the hour's four quarter-hour prices. A desk that bids in the auction needs its forecast before gate closure, so the forecast is issued at 11:00 on d−1 and every input must be public by then. ENTSO-E does not record when each value was published, so each series is taken to be published exactly at its legal deadline.

*Publication times around the forecast, Warsaw time: the legal deadlines of the ENTSO-E series under EU Regulation 543/2013, and the result times in EEX's auction reports.*

| Time | Published | At 11:00 on d−1 |
|---|---|---|
| 11:00 to 11:04 on d−2 | Result of an EUA auction held on d−2 (EEX) | Public: the carbon variants' input |
| 13:00 on d−2 | Prices for d−1 (Art. 12(2)(d)) | Public |
| 18:00 on d−2 | Wind and solar forecast for d−1 (Art. 14(2)(d)) | Public: the main models' renewables input |
| 10:00 on d−1 | Load forecast for d (Art. 6(2)(b)) | Public |
| **11:00 on d−1** | **The forecast is issued** | |
| 11:00 to 11:04 on d−1 | Result of an EUA auction held on d−1 (EEX) | Not yet |
| 12:00 on d−1 | Gate closure for d | |
| 13:00 on d−1 | Prices for d | Not yet: the target |
| 18:00 on d−1 | Wind and solar forecast for d (Art. 14(2)(d)) | Not yet: used only by the papers' setup |

The papers this project follows give their models the wind and solar forecast for day d and treat it as known on d−1. Its deadline is after gate closure, so the main models use the forecast for d−1. The same models given day d's forecast, the papers' setup, measure what that forecast would be worth if it were public by 11:00.

## Models and backtest

The weekly naive copies the price of the same hour one week earlier. The similar-day naive copies the day before from Tuesday to Friday, and the same day one week earlier for Saturday, Sunday and Monday. The linear expert model fits one least-squares regression for each hour on the same hour's price one, two and seven days earlier, the previous day's last, lowest and highest price, the load and the wind and solar forecasts, and weekday and holiday dummies. The gradient-boosted trees (scikit-learn) take the same inputs plus the hour, in one model for all hours. The inputs and models follow [Ziel and Weron (2018)](https://arxiv.org/abs/1805.06649), [Maciejowska, Uniejewski and Weron (2022)](https://arxiv.org/abs/2204.11735) and [Lago et al. (2021)](https://arxiv.org/abs/2008.08004).

For each delivery day from 1 January 2024, both models are retrained on the 365 days before it, so every forecast is out of sample. A Diebold-Mariano test on daily MAE checks whether one model is more accurate than another by more than chance would produce. [METHODOLOGY.md](METHODOLOGY.md) gives the details: the assumptions about the data, the formulas, and where this project departs from the papers.

## Results

Every table, including results by period, year, month and local hour and the full Diebold-Mariano table, is in [`reports/results.md`](reports/results.md).

### Main models

*MAE on the 24,671 hours every model forecast, 1 January 2024 to 24 October 2026. Relative MAE divides a model's MAE by the weekly naive's on the same hours.*

| Model | MAE (EUR/MWh) | Relative MAE vs weekly naive | MAE change vs similar-day naive |
|---|---|---|---|
| Gradient-boosted trees (`gbm`) | 21.18 | 0.65 | −22.5% |
| Linear expert model (`linear`) | 21.30 | 0.65 | −22.1% |
| Similar-day naive (`naive_similar_day`) | 27.34 | 0.84 | 0% |
| Weekly naive (`naive_weekly`) | 32.56 | 1.00 | +19.1% |

- Both models cut the error by 22% against the similar-day naive, and they are tied. They are the only pair the Diebold-Mariano test cannot separate: the one-sided p-value is 0.273 for the trees being more accurate and 0.727 for the linear model, while every other pair differs with p < 0.001.
- The trees lead in 2024 (18.59 against 19.09 EUR/MWh) and trail slightly in 2025 and 2026.
- The weekly naive's MAE rose from 29.86 to 36.99 EUR/MWh after 1 October 2025, and every other model's error rose with it. The trees' relative MAE was 0.65 in both periods, so their error grew in step with the benchmark's. The split cannot separate the switch to 15-minute products from 2026 being a more volatile year.

### Variants

Each variant changes one thing about a main model and is scored on the same hours.

*Each variant against its main model, MAE in EUR/MWh. Differences are computed before rounding, and p is from a two-sided Diebold-Mariano test.*

| Variant | What changes | MAE of the main model | MAE of the variant | Difference | p |
|---|---|---|---|---|---|
| `linear_papers` | The renewables forecast for day d, as in the papers | 21.30 | 18.30 | −3.01 | <0.001 |
| `gbm_papers` | The renewables forecast for day d, as in the papers | 21.18 | 16.95 | −4.23 | <0.001 |
| `linear_asinh` | Fitted on asinh-transformed prices | 21.30 | 20.67 | −0.63 | <0.001 |
| `gbm_defaults` | scikit-learn's default settings | 21.18 | 21.30 | +0.11 | 0.125 |
| `linear_eua` | The carbon price added | 21.30 | 21.28 | −0.02 | 0.810 |
| `gbm_eua` | The carbon price added | 21.18 | 21.12 | −0.06 | 0.529 |

- Day d's renewables forecast would cut the error by 14 to 20%, if it were public by 11:00: the linear model's MAE falls by 3.01 EUR/MWh and the trees' by 4.23 (both p < 0.001). I would expect a desk with its own weather-based forecast for day d to land between the main models and the papers' setup.
- The tree settings barely matter: with scikit-learn's defaults the trees' MAE is 0.11 EUR/MWh higher, and the difference is not significant (p = 0.125).
- Fitting the linear model on asinh-transformed prices lowers its MAE by 3.0% (p < 0.001), because the transform damps the pull of spikes on the least-squares fit. I keep raw prices in the main model, because adopting the transform after seeing it win on the test period would choose the model on the test data.
- The carbon price makes no significant difference (p = 0.810 for the linear model and 0.529 for the trees): it lowers the linear model's MAE by 0.02 EUR/MWh and the trees' by 0.06. A likely reason is that yesterday's prices already carry the cost of carbon into the forecast.

### Where the forecast fails

The error analysis is in [notebook 02](notebooks/02_error_analysis.ipynb).

![Share of hours and of error by price band](reports/figures/error_by_price_level.png)

*Share of hours and share of the trees' error by actual price band. From `notebooks/02_error_analysis.ipynb`.*

- Hours below 0 or at 250 EUR/MWh and above are 5.3% of all hours but carry 17.7% of the trees' error. The forecasts there are far too moderate, partly because sorting hours by the actual price puts in those bands the hours where something unforeseen moved the price.
- The ten worst days, the worst 1% of the 1,028 by the trees' daily MAE, split evenly between under- and over-forecasts. On five, prices were high all day. On the other five the price fell, four of them with strong wind, and on those five the papers' setup averaged an MAE of 44.8 EUR/MWh against 67.2 for the trees. On two of the under-forecast days, 12 and 13 January 2026, Poland peaked 255 and 240 EUR/MWh above both Germany and Czechia.
- The trees' daily bias follows the renewables forecast they could not see. Over all 1,028 days its Spearman rank correlation with the change from the forecast they saw to day d's own is 0.59, against 0.19 for the papers' setup (1 would mean the bias always rises with the change). On nine of the ten worst days the bias had the same sign as the change.

## Limitations

Two assumptions about the data can be wrong in both directions: that each series is published exactly at its deadline, and that the stored values are the ones public at the time. The stored load forecast may be a later update than the version public at 11:00, which would flatter the models. Day d's renewables forecast may have been public before 11:00, which would put the papers' setup within reach. The archive cannot settle either question.

The main models see only Polish electricity data, with no fuel, carbon, flow or outage data, so they react to a cost shock only once it shows up in prices; of these inputs, only carbon is tested. Trees cannot forecast outside the price range of their training window; the linear model can, but in this backtest it also under-forecast spikes heavily: in hours at 250 EUR/MWh and above its bias was −92.9 EUR/MWh, against −104.6 for the trees. The target is hourly, while the auction has traded 15-minute products since October 2025, and the results cover one market and one period, 2024 to 2026.

The forecast is a point forecast, one number per hour with no range of likely prices. It is scored on accuracy alone, and what it would be worth to a desk in money, for example as bids in the auction, is not measured.

## Repository structure

```
METHODOLOGY.md                         the data, publication times, models and evaluation in detail, with the formulas
src/pl_dayahead/
  ingest/entsoe.py   download from ENTSO-E, one month at a time, into entsoe_latest
  ingest/eex.py      download EEX's EU ETS auction results into eua_auctions
  store.py           the DuckDB database: the ENTSO-E series, one row per series and time, the auctions and the forecasts
  staging.py         one row per UTC hour in the hourly table
  timeutils.py       delivery days, clock changes, Polish public holidays, the 11:00 issue time
  availability.py    publication rules and the timing check
  features.py        model inputs, each checked by availability.py
  models/            naive.py, linear.py, boosting.py, and the list of models and variants
  backtest.py        daily walk-forward backtest into the predictions table
  evaluate.py        MAE, relative MAE, Diebold-Mariano tests, reports/results.md, the headline chart
  slopes.py          the linear model's latest coefficients
  style.py           the colours, chart settings and table captions shared by evaluate.py and the notebooks
notebooks/
  01_exploratory_data_analysis.ipynb   price level, intraday profile, negative prices and the supply curve
  02_error_analysis.ipynb              where the forecasts fail: error by time and price level, the worst days, drivers
tests/                                 pytest suite, 131 tests
```

## Reproducing the results

Requirements: Python 3.13 (the version used here), [uv](https://docs.astral.sh/uv/) and a free ENTSO-E API token (register on the Transparency Platform and request RESTful API access, as its [knowledge base](https://transparencyplatform.zendesk.com/) describes). Put the token in a file named `.env` in the project root:

```
ENTSOE_API_TOKEN=your-token
```

Then, from the project root:

```
uv sync
uv run python -m pl_dayahead.ingest.entsoe
uv run python -m pl_dayahead.ingest.eex
uv run python -m pl_dayahead.staging
uv run python -m pl_dayahead.backtest
uv run python -m pl_dayahead.evaluate
uv run pytest
```

The EEX reports need no token. The data is stored in `data/pl_power.duckdb` and the downloaded reports in `data/eex`, neither of which is part of the repository. The backtest runs every model and variant in about 35 minutes; the four tree models take most of that time, because each fits 1,028 models. Open the notebooks with `uv run jupyter lab`.

## License

MIT. See [LICENSE](LICENSE).
