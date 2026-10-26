"""The forecasting models.

Every model has the same members, so the backtest can treat them all alike:

    name         the label stored with its forecasts
    features     the input columns it uses
    fit(X, y)    learn from the training rows
    predict(X)   return one forecast per row

To add a model, write a class with these members and add it to MODELS.
"""

from pl_dayahead.models.boosting import GradientBoosting, GradientBoostingResD1
from pl_dayahead.models.linear import LinearByHour, LinearResD1
from pl_dayahead.models.naive import NaiveSimilarDay, NaiveWeekly

MODELS = {
    model.name: model
    for model in [NaiveWeekly, NaiveSimilarDay, LinearByHour, LinearResD1, GradientBoosting, GradientBoostingResD1]
}
