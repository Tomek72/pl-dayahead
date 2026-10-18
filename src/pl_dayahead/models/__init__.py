"""Forecasting models.

Every model has the same small interface, so the backtest treats them identically:

    name: str                    label stored with its predictions
    features: list[str]          feature-table columns the model uses
    public_at_issue_time: bool   True if every input must have been public at 11:00 on d-1;
                                 backtest.py refuses to run such a model if an input relies
                                 on the papers' convention (see availability.py)
    fit(X, y) -> self            learn from training rows (X holds only `features`)
    predict(X) -> array          one price per row of X

A new model is a class with these members, registered in MODELS.
"""

from pl_dayahead.models.boosting import GradientBoosting, GradientBoostingResD1
from pl_dayahead.models.linear import LinearByHour, LinearResD1
from pl_dayahead.models.naive import NaiveSimilarDay, NaiveWeekly

MODELS = {
    model.name: model
    for model in [NaiveWeekly, NaiveSimilarDay, LinearByHour, LinearResD1, GradientBoosting, GradientBoostingResD1]
}
