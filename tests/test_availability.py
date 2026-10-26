import pytest

from pl_dayahead.availability import AvailabilityError, FeatureSpec, check, in_time
from pl_dayahead.features import FEATURES, SPECS
from pl_dayahead.models import MODELS


def test_target_day_price_is_never_in_time():
    assert not in_time(FeatureSpec("p", "price", 0))  # due 13:00, after the 11:00 forecast


def test_earlier_prices_are_in_time():
    assert in_time(FeatureSpec("p", "price", -1))
    assert in_time(FeatureSpec("p", "price", -7))


def test_target_day_load_forecast_is_in_time():
    assert in_time(FeatureSpec("l", "load_fcst", 0))  # due 10:00


def test_renewables_for_the_target_day_are_late_and_for_the_day_before_in_time():
    assert not in_time(FeatureSpec("r", "res_fcst", 0))  # due 18:00
    assert in_time(FeatureSpec("r", "res_fcst", -1))


def test_project_features_pass_with_the_convention_reported():
    assert check(SPECS) == ["res_fcst"]


def test_a_feature_using_the_target_day_price_is_rejected():
    with pytest.raises(AvailabilityError, match="leak"):
        check([FeatureSpec("leak", "price", 0)])


def test_the_target_day_maximum_price_is_rejected():
    with pytest.raises(AvailabilityError, match="leak_max"):
        check([FeatureSpec("leak_max", "price", 0, "max")])


def test_the_convention_covers_the_target_day_only():
    with pytest.raises(AvailabilityError, match="res_tomorrow"):
        check([FeatureSpec("res_tomorrow", "res_fcst", 1)])


def test_a_series_without_a_rule_is_rejected():
    with pytest.raises(AvailabilityError, match="no publication rule"):
        check([FeatureSpec("mystery", "gas_price", -1)])


def test_an_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError, match="agg"):
        FeatureSpec("odd", "price", -1, "median")


def test_every_registered_model_uses_only_known_features():
    for name, model in MODELS.items():
        assert set(model.features) <= set(FEATURES), name


def test_only_the_main_linear_and_tree_models_use_the_target_day_renewables():
    relying = sorted(name for name, model in MODELS.items() if "res_fcst" in model.features)
    assert relying == ["gbm", "linear"]