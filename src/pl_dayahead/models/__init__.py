"""The forecasting models.

Every model has the same members, so the backtest can treat them all alike:

    name         the label stored with its forecasts
    features     the input columns it uses
    fit(X, y)    learn from the training rows
    predict(X)   return one forecast per row

A variant also has `base`: the name of the main model it changes in one respect. The variants are
the papers' setup, which uses a forecast published after 11:00, and two robustness checks.
evaluate.py compares each variant with its base model, apart from the main tables.

To add a model, write a class with these members and add it to MODELS.
"""

from pl_dayahead.models.boosting import GradientBoosting, GradientBoostingDefaults, GradientBoostingPapers
from pl_dayahead.models.linear import LinearAsinh, LinearByHour, LinearPapers
from pl_dayahead.models.naive import NaiveSimilarDay, NaiveWeekly

MODELS = {
    model.name: model
    for model in [NaiveWeekly, NaiveSimilarDay, LinearByHour, GradientBoosting,
                  LinearPapers, GradientBoostingPapers, LinearAsinh, GradientBoostingDefaults]
}
VARIANTS = {name: model.base for name, model in MODELS.items() if hasattr(model, "base")}  # variant -> base
