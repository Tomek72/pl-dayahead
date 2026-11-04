"""The forecasting models.

Every model has the same members, so the backtest can treat them all alike:

    name         the label stored with its forecasts
    features     the input columns it uses
    fit(X, y)    learn from the training rows
    predict(X)   return one forecast per row

A variant also has `base`: the name of the main model it changes in one respect. The variants are
the papers' setup, which uses a forecast published after 11:00, two robustness checks, and the
carbon variants, which add the price of EU emission allowances. evaluate.py compares each variant
with its base model, apart from the main tables. The papers' setup also has `papers_setup = True`,
which makes it the only kind of model the backtest lets use an input published after 11:00.

To add a model, write a class with these members and add it to MODELS.
"""

from pl_dayahead.models.boosting import (GradientBoosting, GradientBoostingCarbon, GradientBoostingDefaults,
                                         GradientBoostingPapers)
from pl_dayahead.models.linear import LinearAsinh, LinearByHour, LinearCarbon, LinearPapers
from pl_dayahead.models.naive import NaiveSimilarDay, NaiveWeekly

MODELS = {
    model.name: model
    for model in [NaiveWeekly, NaiveSimilarDay, LinearByHour, GradientBoosting,
                  LinearPapers, GradientBoostingPapers, LinearAsinh, GradientBoostingDefaults,
                  LinearCarbon, GradientBoostingCarbon]
}
VARIANTS = {name: model.base for name, model in MODELS.items() if hasattr(model, "base")}  # variant -> base
# main model -> its papers' setup
PAPERS_SETUP = {model.base: name for name, model in MODELS.items() if getattr(model, "papers_setup", False)}
