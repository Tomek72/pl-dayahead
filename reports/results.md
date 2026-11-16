# Backtest results

MAE in EUR/MWh on 24,671 hours forecast by every stored registered model, 1 Jan 2024 to 24 Oct 2026. The common-sample restriction excludes 0 stored hours.

## Overall

|  | MAE naive_weekly | MAE gbm | MAE linear | MAE naive_similar_day | rel MAE gbm | rel MAE linear | rel MAE naive_similar_day | hours | % vs naive_similar_day naive_weekly | % vs naive_similar_day gbm | % vs naive_similar_day linear |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all hours | 32.56 | 21.18 | 21.30 | 27.34 | 0.65 | 0.65 | 0.84 | 24671 | 19.10 | -22.52 | -22.07 |

## By period

| period | MAE naive_weekly | MAE gbm | MAE linear | MAE naive_similar_day | rel MAE gbm | rel MAE linear | rel MAE naive_similar_day | hours |
|---|---|---|---|---|---|---|---|---|
| hourly MTU (to Sep 2025) | 29.86 | 19.45 | 19.66 | 24.56 | 0.65 | 0.66 | 0.82 | 15335 |
| 15-minute MTU (from Oct 2025) | 36.99 | 24.02 | 24.00 | 31.91 | 0.65 | 0.65 | 0.86 | 9336 |

## By year

| year | MAE naive_weekly | MAE gbm | MAE linear | MAE naive_similar_day | rel MAE gbm | rel MAE linear | rel MAE naive_similar_day | hours |
|---|---|---|---|---|---|---|---|---|
| 2024 | 29.85 | 18.59 | 19.09 | 24.10 | 0.62 | 0.64 | 0.81 | 8784 |
| 2025 | 30.02 | 20.36 | 20.24 | 25.59 | 0.68 | 0.67 | 0.85 | 8760 |
| 2026 | 39.02 | 25.39 | 25.35 | 33.48 | 0.65 | 0.65 | 0.86 | 7127 |

## By month

| month | MAE naive_weekly | MAE gbm | MAE linear | MAE naive_similar_day | rel MAE gbm | rel MAE linear | rel MAE naive_similar_day | hours |
|---|---|---|---|---|---|---|---|---|
| 2024-01 | 29.08 | 12.29 | 13.49 | 19.03 | 0.42 | 0.46 | 0.65 | 744 |
| 2024-02 | 18.34 | 9.91 | 10.41 | 15.01 | 0.54 | 0.57 | 0.82 | 696 |
| 2024-03 | 13.17 | 9.59 | 9.92 | 11.61 | 0.73 | 0.75 | 0.88 | 743 |
| 2024-04 | 20.43 | 12.28 | 12.13 | 19.55 | 0.60 | 0.59 | 0.96 | 720 |
| 2024-05 | 25.21 | 13.68 | 13.72 | 14.24 | 0.54 | 0.54 | 0.56 | 744 |
| 2024-06 | 25.22 | 22.92 | 22.90 | 25.26 | 0.91 | 0.91 | 1.00 | 720 |
| 2024-07 | 36.89 | 28.05 | 27.62 | 29.28 | 0.76 | 0.75 | 0.79 | 744 |
| 2024-08 | 31.75 | 20.89 | 20.80 | 27.31 | 0.66 | 0.66 | 0.86 | 744 |
| 2024-09 | 32.55 | 21.98 | 21.94 | 26.50 | 0.68 | 0.67 | 0.81 | 720 |
| 2024-10 | 39.92 | 23.92 | 24.08 | 32.21 | 0.60 | 0.60 | 0.81 | 745 |
| 2024-11 | 38.98 | 25.76 | 28.32 | 33.20 | 0.66 | 0.73 | 0.85 | 720 |
| 2024-12 | 45.79 | 21.52 | 23.45 | 35.64 | 0.47 | 0.51 | 0.78 | 744 |
| 2025-01 | 38.59 | 24.29 | 22.87 | 30.00 | 0.63 | 0.59 | 0.78 | 744 |
| 2025-02 | 30.36 | 19.22 | 18.88 | 23.45 | 0.63 | 0.62 | 0.77 | 672 |
| 2025-03 | 36.95 | 21.14 | 21.10 | 25.17 | 0.57 | 0.57 | 0.68 | 743 |
| 2025-04 | 26.48 | 20.78 | 19.49 | 22.01 | 0.78 | 0.74 | 0.83 | 720 |
| 2025-05 | 28.84 | 20.45 | 21.10 | 25.82 | 0.71 | 0.73 | 0.90 | 744 |
| 2025-06 | 31.72 | 24.01 | 24.18 | 32.45 | 0.76 | 0.76 | 1.02 | 720 |
| 2025-07 | 24.51 | 17.00 | 18.01 | 20.01 | 0.69 | 0.73 | 0.82 | 744 |
| 2025-08 | 21.55 | 14.85 | 15.52 | 20.10 | 0.69 | 0.72 | 0.93 | 744 |
| 2025-09 | 29.92 | 23.78 | 22.78 | 27.55 | 0.79 | 0.76 | 0.92 | 720 |
| 2025-10 | 33.88 | 19.82 | 19.85 | 30.97 | 0.59 | 0.59 | 0.91 | 745 |
| 2025-11 | 30.40 | 22.82 | 21.77 | 30.33 | 0.75 | 0.72 | 1.00 | 720 |
| 2025-12 | 27.05 | 16.34 | 17.41 | 19.31 | 0.60 | 0.64 | 0.71 | 744 |
| 2026-01 | 51.41 | 29.19 | 31.93 | 42.38 | 0.57 | 0.62 | 0.82 | 744 |
| 2026-02 | 23.59 | 16.90 | 18.41 | 17.22 | 0.72 | 0.78 | 0.73 | 672 |
| 2026-03 | 34.39 | 22.88 | 21.35 | 30.99 | 0.67 | 0.62 | 0.90 | 743 |
| 2026-04 | 44.33 | 26.64 | 25.54 | 38.90 | 0.60 | 0.58 | 0.88 | 720 |
| 2026-05 | 35.35 | 20.35 | 19.31 | 28.29 | 0.58 | 0.55 | 0.80 | 744 |
| 2026-06 | 34.11 | 23.79 | 24.16 | 29.66 | 0.70 | 0.71 | 0.87 | 720 |
| 2026-07 | 35.69 | 22.38 | 21.61 | 25.47 | 0.63 | 0.61 | 0.71 | 744 |
| 2026-08 | 35.93 | 25.51 | 25.62 | 30.48 | 0.71 | 0.71 | 0.85 | 744 |
| 2026-09 | 49.53 | 32.33 | 31.53 | 47.68 | 0.65 | 0.64 | 0.96 | 720 |
| 2026-10 | 46.41 | 35.59 | 35.88 | 45.41 | 0.77 | 0.77 | 0.98 | 576 |

## By local hour

| local_hour | MAE naive_weekly | MAE gbm | MAE linear | MAE naive_similar_day | rel MAE gbm | rel MAE linear | rel MAE naive_similar_day | hours |
|---|---|---|---|---|---|---|---|---|
| 0 | 20.92 | 10.38 | 8.54 | 18.47 | 0.50 | 0.41 | 0.88 | 1028 |
| 1 | 19.99 | 10.09 | 9.14 | 17.65 | 0.50 | 0.46 | 0.88 | 1028 |
| 2 | 20.24 | 10.52 | 10.01 | 17.98 | 0.52 | 0.49 | 0.89 | 1027 |
| 3 | 20.01 | 10.44 | 10.19 | 17.80 | 0.52 | 0.51 | 0.89 | 1028 |
| 4 | 19.66 | 10.41 | 10.33 | 17.40 | 0.53 | 0.53 | 0.88 | 1028 |
| 5 | 19.48 | 10.75 | 10.37 | 16.96 | 0.55 | 0.53 | 0.87 | 1028 |
| 6 | 24.15 | 15.70 | 15.02 | 21.07 | 0.65 | 0.62 | 0.87 | 1028 |
| 7 | 31.31 | 20.61 | 21.42 | 26.61 | 0.66 | 0.68 | 0.85 | 1028 |
| 8 | 33.31 | 22.65 | 23.33 | 28.51 | 0.68 | 0.70 | 0.86 | 1028 |
| 9 | 35.70 | 23.87 | 24.22 | 30.00 | 0.67 | 0.68 | 0.84 | 1028 |
| 10 | 38.63 | 25.54 | 25.94 | 31.10 | 0.66 | 0.67 | 0.81 | 1028 |
| 11 | 42.75 | 27.39 | 28.32 | 34.34 | 0.64 | 0.66 | 0.80 | 1028 |
| 12 | 45.46 | 29.67 | 29.83 | 36.54 | 0.65 | 0.66 | 0.80 | 1028 |
| 13 | 46.25 | 30.29 | 30.65 | 37.77 | 0.65 | 0.66 | 0.82 | 1028 |
| 14 | 44.76 | 29.09 | 29.98 | 36.47 | 0.65 | 0.67 | 0.81 | 1028 |
| 15 | 41.55 | 26.61 | 28.17 | 34.43 | 0.64 | 0.68 | 0.83 | 1028 |
| 16 | 38.93 | 26.09 | 27.26 | 33.13 | 0.67 | 0.70 | 0.85 | 1028 |
| 17 | 36.96 | 25.85 | 27.40 | 31.90 | 0.70 | 0.74 | 0.86 | 1028 |
| 18 | 39.35 | 27.10 | 28.14 | 33.29 | 0.69 | 0.72 | 0.85 | 1028 |
| 19 | 47.91 | 34.16 | 33.34 | 39.82 | 0.71 | 0.70 | 0.83 | 1028 |
| 20 | 45.22 | 32.23 | 31.79 | 37.36 | 0.71 | 0.70 | 0.83 | 1028 |
| 21 | 30.14 | 21.56 | 21.33 | 25.21 | 0.72 | 0.71 | 0.84 | 1028 |
| 22 | 20.12 | 14.37 | 13.82 | 16.80 | 0.71 | 0.69 | 0.84 | 1028 |
| 23 | 18.66 | 12.97 | 12.76 | 15.52 | 0.69 | 0.68 | 0.83 | 1028 |

## Main-model Diebold-Mariano comparisons

Two-sided tests of equal predictive accuracy on daily MAE over 1,028 days. Difference is Model B minus Model A in EUR/MWh; positive values favour Model A. The raw p-values use a Newey-West HAC standard error with 7 Bartlett lags; Holm adjustment covers the 6 available unique pairs. Significance threshold: 5%. P-values are shown at full precision. Comparisons with fewer than two days or non-positive estimated variance are marked not testable. They remain in the adjustment family with p=1 internally; no test result is claimed for them.

| Model A | Model B | Mean daily MAE difference (B − A) | Raw two-sided p-value | Holm-adjusted p-value | Significant after adjustment |
|---|---|---|---|---|---|
| gbm | linear | 0.12 | 0.5468410563489451 | 0.5468410563489451 | No |
| gbm | naive_similar_day | 6.16 | 1.9675796743076675e-30 | 7.87031869723067e-30 | Yes |
| gbm | naive_weekly | 11.38 | 6.020802448035644e-57 | 3.6124814688213862e-56 | Yes |
| linear | naive_similar_day | 6.03 | 8.507626316236101e-27 | 2.5522878948708302e-26 | Yes |
| linear | naive_weekly | 11.26 | 7.992881793787455e-53 | 3.9964408968937274e-52 | Yes |
| naive_similar_day | naive_weekly | 5.22 | 1.3297810441302367e-19 | 2.6595620882604735e-19 | Yes |

## Exploratory variant comparisons

Each variant changes one input or setting in its corresponding main model. The mean daily MAE difference is variant minus main model; negative values favour the variant. Two-sided DM p-values use the seven-lag HAC standard error. Holm adjustment is a separate family across 6 of 6 registered comparisons available in this report; significance threshold: 5%. These are secondary sensitivities. Multiplicity adjustment does not remove the limitations of selecting or evaluating variants on the same test period. Untestable comparisons follow the same convention as the main-model family.

| variant | Corresponding main model | Mean daily MAE difference (variant − main) | Raw two-sided p-value | Holm-adjusted p-value | Significant after adjustment |
|---|---|---|---|---|---|
| linear_papers | linear | -3.01 | 1.5662511385767302e-25 | 7.83125569288365e-25 | Yes |
| gbm_papers | gbm | -4.23 | 2.284332532035016e-52 | 1.3705995192210096e-51 | Yes |
| linear_asinh | linear | -0.63 | 6.599644022697632e-06 | 2.6398576090790527e-05 | Yes |
| gbm_defaults | gbm | 0.11 | 0.12507295900257467 | 0.37521887700772405 | No |
| linear_eua | linear | -0.02 | 0.8102586271408022 | 1.0 | No |
| gbm_eua | gbm | -0.06 | 0.5288242184106915 | 1.0 | No |

## HAC bandwidth robustness diagnostic

Sample autocorrelations of the 6 available main-model daily loss-difference series at lags 8–28 range from 0.057 to 0.119 for the pair-specific largest absolute coefficient shown below (signed values). The table reports two-sided raw p-values with Bartlett Newey-West bandwidths of 7, 14 and 28 days. This is a robustness diagnostic, not an additional testing family; the seven-lag specification remains primary. The table permits comparison of the raw p-values across bandwidths.

| Model A | Model B | Largest absolute ACF, lags 8–28 | Lag |
|---|---|---|---|
| gbm | linear | -0.065 | 25 |
| gbm | naive_similar_day | 0.079 | 21 |
| gbm | naive_weekly | 0.085 | 26 |
| linear | naive_similar_day | -0.057 | 24 |
| linear | naive_weekly | 0.075 | 26 |
| naive_similar_day | naive_weekly | 0.119 | 14 |

| Model A | Model B | Raw p, 7 lags | Raw p, 14 lags | Raw p, 28 lags |
|---|---|---|---|---|
| gbm | linear | 0.5468410563489451 | 0.54041536553632 | 0.5401224271664118 |
| gbm | naive_similar_day | 1.9675796743076675e-30 | 8.27522917950206e-26 | 3.512400209483208e-21 |
| gbm | naive_weekly | 6.020802448035644e-57 | 3.257776113356557e-48 | 6.001096661229331e-41 |
| linear | naive_similar_day | 8.507626316236101e-27 | 8.290360650865768e-23 | 1.0861879987088231e-19 |
| linear | naive_weekly | 7.992881793787455e-53 | 9.043179748880472e-46 | 1.4910930365104597e-40 |
| naive_similar_day | naive_weekly | 1.3297810441302367e-19 | 5.8993145240680836e-18 | 6.647333391474091e-17 |
