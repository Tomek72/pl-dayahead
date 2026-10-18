from datetime import date, timedelta

import pandas as pd
import pytest

from pl_dayahead.availability import AvailabilityError, FeatureSpec, available_at, check, check_model
from pl_dayahead.features import FEATURES, SPECS
from pl_dayahead.models import MODELS
from pl_dayahead.models.linear import LinearByHour, LinearResD1
from pl_dayahead.timeutils import issue_time_utc

TARGET = date(2026, 1, 15)
DAYS = [date(2025, 3, 29) + timedelta(days=i) for i in range(400)]  # covers both clock changes


def test_publication_times_in_utc():
    # Prices for 15 January are public at 13:00 Warsaw (CET) on 14 January.
    assert available_at("price", TARGET) == pd.Timestamp("2026-01-14 12:00", tz="UTC")
    # After the spring change Warsaw is UTC+2, so 13:00 local is 11:00 UTC.
    assert available_at("price", date(2026, 3, 30)) == pd.Timestamp("2026-03-29 11:00", tz="UTC")


def test_target_day_price_is_never_available():
    assert available_at("price", TARGET) > issue_time_utc(TARGET)


def test_previous_day_price_is_available():
    assert available_at("price", TARGET - timedelta(days=1)) <= issue_time_utc(TARGET)


def test_load_forecast_for_target_day_is_available():
    assert available_at("load_fcst", TARGET) <= issue_time_utc(TARGET)


def test_renewables_for_the_target_day_are_late_and_for_the_day_before_on_time():
    assert available_at("res_fcst", TARGET) > issue_time_utc(TARGET)
    assert available_at("res_fcst", TARGET - timedelta(days=1)) <= issue_time_utc(TARGET)


def test_project_features_pass_with_the_convention_reported():
    assert check(SPECS, DAYS) == ["res_fcst"]


def test_a_feature_using_the_target_day_price_is_rejected():
    with pytest.raises(AvailabilityError, match="leak"):
        check([FeatureSpec("leak", "price", 0)], DAYS)


def test_the_target_day_maximum_price_is_rejected():
    with pytest.raises(AvailabilityError, match="leak_max"):
        check([FeatureSpec("leak_max", "price", 0, "max")], DAYS)


def test_the_convention_covers_the_target_day_only():
    with pytest.raises(AvailabilityError, match="res_tomorrow"):
        check([FeatureSpec("res_tomorrow", "res_fcst", 1)], DAYS)


def test_a_series_without_a_rule_is_rejected():
    with pytest.raises(AvailabilityError, match="no availability rule"):
        check([FeatureSpec("mystery", "gas_price", -1)], DAYS)


def test_an_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="agg"):
        FeatureSpec("odd", "price", -1, "median")


def test_a_comparison_model_using_the_day_d_forecast_is_refused():
    with pytest.raises(AvailabilityError, match="res_fcst"):
        check_model("bad", LinearByHour.features, ["res_fcst"], public_at_issue_time=True)


def test_main_and_comparison_linear_models_pass():
    check_model(LinearByHour.name, LinearByHour.features, ["res_fcst"], LinearByHour.public_at_issue_time)
    check_model(LinearResD1.name, LinearResD1.features, ["res_fcst"], LinearResD1.public_at_issue_time)


def test_every_registered_model_passes_and_uses_only_known_features():
    convention = check(SPECS, DAYS)
    for name, model in MODELS.items():
        check_model(name, model.features, convention, model.public_at_issue_time)
        assert set(model.features) <= set(FEATURES), name


def test_only_the_main_linear_and_tree_models_rely_on_the_convention():
    relying = sorted(name for name, model in MODELS.items() if "res_fcst" in model.features)
    assert relying == ["gbm", "linear"]
    assert all(not MODELS[name].public_at_issue_time for name in relying)
