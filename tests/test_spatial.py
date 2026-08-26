"""Tests for the physical plausibility checks in src/spatial.py.

Synthetic models with known behaviour are used throughout: one that attenuates
correctly, one that gets the sign backwards, one that oscillates, and one that
is flat. Each should be judged the way a person looking at its shake map would
judge it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import spatial as S
from features import MODEL_FEATURES


class FakeRegressor:
    """A stand-in model that applies a supplied rule to log distance."""

    def __init__(self, rule):
        self.rule = rule

    def predict(self, X):
        return self.rule(np.asarray(X["log_hypocentral_distance"], dtype="float64"))


def sensible_attenuation(log_distance):
    """Intensity falling steadily with distance, as physics requires."""
    return 8.0 - 1.5 * log_distance


def backwards_attenuation(log_distance):
    """Intensity rising with distance. Physically impossible."""
    return 2.0 + 1.5 * log_distance


def oscillating(log_distance):
    """Falls overall but wobbles, which produces a blotchy map."""
    return 8.0 - 1.5 * log_distance + 0.4 * np.sin(log_distance * 25)


def nearly_flat(log_distance):
    """Monotonic but barely varies, so the map says almost nothing."""
    return 4.5 - 0.02 * log_distance


def distant_hotspot(log_distance):
    """Attenuates correctly, then predicts severe shaking at the far edge.

    This is the failure that got through. The sweep used to stop at 400 km, so
    everything this model does past that point went unexamined, and it was
    accepted on the strength of the part that behaved. The maps showed MMI 8
    across Northland for a Kaikoura earthquake.
    """
    return np.where(log_distance > 2.7, 8.0, 8.0 - 1.5 * log_distance)


class DirectionalBlowUp:
    """Attenuates properly in every direction but one.

    Averaging over bearings dilutes this to something that still trends
    downwards, so the shape checks accept it. The map does not.
    """

    def predict(self, X):
        base = 8.0 - 1.5 * X["log_hypocentral_distance"]
        due_north = np.asarray(X["azimuth_cos"]) > 0.99
        far = np.asarray(X["log_hypocentral_distance"]) > 2.7
        return np.where(due_north & far, 8.0, base)


def profile_for(rule):
    return S.radial_profile(FakeRegressor(rule), MODEL_FEATURES, kind="regression")


# --- the checks themselves ---------------------------------------------------

def test_sensible_attenuation_is_recognised():
    result = S.attenuation_check(profile_for(sensible_attenuation))
    assert result["spearman"] < -0.99
    assert result["fraction_decreasing"] == 1.0
    assert result["total_drop"] > 0.5
    assert result["monotonic"]


def test_backwards_attenuation_is_caught():
    """The failure the whole check exists to catch."""
    result = S.attenuation_check(profile_for(backwards_attenuation))
    assert result["spearman"] > 0.99
    assert not result["monotonic"]
    assert result["total_drop"] < 0


def test_oscillation_is_caught_even_though_the_trend_is_right():
    result = S.attenuation_check(profile_for(oscillating))
    assert result["spearman"] < -0.5          # trend is still downward
    assert result["fraction_decreasing"] < 1.0  # but not every step falls
    assert not result["monotonic"]


def test_a_flat_model_is_monotonic_but_says_nothing():
    """Strict monotonicity alone would pass this, which is why drop is checked."""
    result = S.attenuation_check(profile_for(nearly_flat))
    assert result["monotonic"]
    assert result["total_drop"] < S.MINIMUM_TOTAL_DROP


def test_constant_prediction_is_handled():
    result = S.attenuation_check(profile_for(lambda d: np.full_like(d, 5.0)))
    assert result["total_drop"] == 0.0
    assert not result["monotonic"]


# --- verdicts ----------------------------------------------------------------

def test_good_model_passes():
    summary = S.physical_plausibility(FakeRegressor(sensible_attenuation),
                                      MODEL_FEATURES, kind="regression")
    assert S.plausibility_verdict(summary)["passes"]


def test_backwards_model_is_rejected_with_a_reason():
    summary = S.physical_plausibility(FakeRegressor(backwards_attenuation),
                                      MODEL_FEATURES, kind="regression")
    verdict = S.plausibility_verdict(summary)
    assert not verdict["passes"]
    assert any("decay" in reason for reason in verdict["reasons"])


def test_flat_model_is_rejected_for_saying_nothing():
    summary = S.physical_plausibility(FakeRegressor(nearly_flat),
                                      MODEL_FEATURES, kind="regression")
    verdict = S.plausibility_verdict(summary)
    assert not verdict["passes"]
    assert any("varies" in reason for reason in verdict["reasons"])


def test_rejection_reasons_are_specific():
    """A rejection has to be explainable, not just asserted."""
    summary = S.physical_plausibility(FakeRegressor(backwards_attenuation),
                                      MODEL_FEATURES, kind="regression")
    for reason in S.plausibility_verdict(summary)["reasons"]:
        assert any(char.isdigit() for char in reason)


# --- profile construction ----------------------------------------------------

def test_profile_covers_increasing_distances():
    profile = profile_for(sensible_attenuation)
    distances = profile["epicentral_distance_km"].to_numpy()
    assert (np.diff(distances) > 0).all()
    assert len(profile) > 10


def test_profile_averages_over_bearings():
    """One direction must not decide the verdict for a direction-aware model."""
    class DirectionalModel:
        def predict(self, X):
            # Falls to the north, rises to the south
            return 5.0 - 2.0 * X["azimuth_cos"] * X["log_hypocentral_distance"]

    profile = S.radial_profile(DirectionalModel(), MODEL_FEATURES,
                               kind="regression", n_azimuths=8)
    # Averaged over the compass the directional term cancels, leaving a flat
    # profile rather than whichever bearing was sampled first.
    assert profile["predicted_mmi"].std() < 0.1


def test_larger_magnitudes_are_checked_too():
    """A model can behave for moderate events and misbehave for large ones."""
    summary = S.physical_plausibility(FakeRegressor(sensible_attenuation),
                                      MODEL_FEATURES, kind="regression")
    assert len(summary["by_magnitude"]) == 4
    assert set(summary["by_magnitude"]["magnitude"]) == {4.5, 5.5, 6.5, 7.5}


# --- mapping -----------------------------------------------------------------

def test_grid_covers_the_country():
    grid = S.nz_grid(spacing_km=25)
    assert grid["latitude"].min() < -46
    assert grid["latitude"].max() > -35
    assert grid["longitude"].min() < 168
    assert grid["longitude"].max() > 177


def test_shake_map_is_strongest_near_the_epicentre():
    event = {"magnitude": 7.0, "depth_km": 15.0, "longitude": 172.6, "latitude": -43.5}
    mapped = S.shake_map(FakeRegressor(sensible_attenuation), MODEL_FEATURES,
                         event, grid=S.nz_grid(spacing_km=25), kind="regression")

    nearest = mapped.nsmallest(20, "epicentral_distance_km")["predicted_mmi"].mean()
    furthest = mapped.nlargest(20, "epicentral_distance_km")["predicted_mmi"].mean()
    assert nearest > furthest


# --- the far field, which the shape checks alone do not cover -----------------

def test_sweep_covers_the_whole_country():
    """A check that stops short of 900 km leaves part of every map untested."""
    profile = profile_for(sensible_attenuation)
    assert profile["epicentral_distance_km"].max() >= 900


def test_distant_hotspot_would_have_passed_the_old_short_sweep():
    """Documents the bug: truncated at 400 km, this model looks perfect."""
    truncated = S.radial_profile(FakeRegressor(distant_hotspot), MODEL_FEATURES,
                                 distances_km=np.geomspace(5, 400, 40),
                                 kind="regression")
    result = S.attenuation_check(truncated)
    assert result["spearman"] < S.MINIMUM_SPEARMAN
    assert result["total_drop"] > S.MINIMUM_TOTAL_DROP


def test_distant_hotspot_is_caught():
    summary = S.physical_plausibility(FakeRegressor(distant_hotspot),
                                      MODEL_FEATURES, kind="regression")
    verdict = S.plausibility_verdict(summary)
    assert not verdict["passes"]
    assert any("distant hotspot" in reason for reason in verdict["reasons"])


def test_one_bad_bearing_is_caught_even_though_the_average_is_fine():
    """The case bearing-averaging is blind to."""
    model = DirectionalBlowUp()

    averaged = S.attenuation_check(
        S.radial_profile(model, MODEL_FEATURES, kind="regression"))
    assert averaged["spearman"] < S.MINIMUM_SPEARMAN  # the shape check is happy

    verdict = S.plausibility_verdict(
        S.physical_plausibility(model, MODEL_FEATURES, kind="regression"))
    assert not verdict["passes"]
    assert any("distant hotspot" in reason for reason in verdict["reasons"])


def test_a_sound_model_has_no_far_field_excess():
    result = S.far_field_check(FakeRegressor(sensible_attenuation),
                               MODEL_FEATURES, kind="regression")
    assert result["max_far_field"] < result["near_field"]
    assert result["far_field_excess"] < 0


def test_far_field_reason_names_the_numbers():
    summary = S.physical_plausibility(FakeRegressor(distant_hotspot),
                                      MODEL_FEATURES, kind="regression")
    reason = next(r for r in S.plausibility_verdict(summary)["reasons"]
                  if "distant hotspot" in r)
    assert "8.00" in reason and "400 km" in reason
