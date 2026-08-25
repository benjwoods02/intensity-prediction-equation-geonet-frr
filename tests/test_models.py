"""Tests for splitting, baseline and evaluation in src/models.py.

The split tests are the important ones. Leakage between train and test is
silent: it does not raise, it just produces optimistic numbers. These pin the
guarantee that no earthquake can appear on both sides.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import models as M
from clean import apply_weight_scheme


def make_rows(events_and_magnitudes, rows_per_event=10):
    """Build a frame of cells spread across several earthquakes."""
    records = []
    for public_id, magnitude in events_and_magnitudes:
        for index in range(rows_per_event):
            records.append({
                "public_id": public_id,
                "cell_x": index,
                "cell_y": index,
                "magnitude": magnitude,
                "log_hypocentral_distance": 1.0 + index / 10,
                "mmi": 4,
                "mmi_mean": 4.0,
                "weight": 1.0,
            })
    return pd.DataFrame(records)


def many_events(count=40):
    return [(f"e{i}", 4.0 + (i % 8) * 0.5) for i in range(count)]


# --- splitting ---------------------------------------------------------------

def test_no_earthquake_appears_on_both_sides():
    """The whole point of the split. Leakage here is silent and inflates scores."""
    frame = make_rows(many_events())
    train, test = M.split_by_event(frame, verbose=False)
    assert set(train["public_id"]).isdisjoint(set(test["public_id"]))


def test_every_row_lands_somewhere():
    frame = make_rows(many_events())
    train, test = M.split_by_event(frame, verbose=False)
    assert len(train) + len(test) == len(frame)


def test_split_is_reproducible():
    frame = make_rows(many_events())
    first, _ = M.split_by_event(frame, random_state=7, verbose=False)
    second, _ = M.split_by_event(frame, random_state=7, verbose=False)
    assert sorted(first["public_id"].unique()) == sorted(second["public_id"].unique())


def test_a_lone_large_event_does_not_break_stratification():
    """There is one event above magnitude 7.5 in the real catalogue.

    A stratum with a single member cannot be split, so sparse high bands have
    to be folded down rather than causing a failure.
    """
    events = [(f"small{i}", 4.5) for i in range(10)] + [("huge", 7.8)]
    frame = make_rows(events)
    train, test = M.split_by_event(frame, verbose=False)
    assert set(train["public_id"]).isdisjoint(set(test["public_id"]))


def test_stratification_spreads_magnitudes_across_both_sides():
    frame = make_rows(many_events(48))
    train, test = M.split_by_event(frame, test_size=0.3, verbose=False)
    # Both sides should span more than a single magnitude band
    assert train["magnitude"].nunique() > 1
    assert test["magnitude"].nunique() > 1


def test_folds_never_share_an_earthquake():
    frame = make_rows(many_events())
    for train_mask, test_mask in M.event_folds(frame, n_splits=3):
        train_events = set(frame.loc[train_mask, "public_id"])
        test_events = set(frame.loc[test_mask, "public_id"])
        assert train_events.isdisjoint(test_events)


def test_folds_cover_every_row_exactly_once_as_test():
    frame = make_rows(many_events())
    seen = np.zeros(len(frame), dtype=int)
    for _, test_mask in M.event_folds(frame, n_splits=5):
        seen += test_mask.astype(int)
    assert (seen == 1).all()


# --- baseline ----------------------------------------------------------------

def test_baseline_recovers_a_known_relationship():
    """Fit against data generated from an exact equation and check it comes back."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "magnitude": rng.uniform(4, 8, 500),
        "log_hypocentral_distance": rng.uniform(1, 3, 500),
    })
    target = 2.0 + 1.5 * frame["magnitude"] - 2.5 * frame["log_hypocentral_distance"]

    model = M.AttenuationBaseline().fit(frame, target)

    assert model.coefficients_["magnitude"] == pytest.approx(1.5, abs=0.01)
    assert model.coefficients_["log_distance"] == pytest.approx(-2.5, abs=0.01)


def test_baseline_equation_is_readable():
    frame = pd.DataFrame({"magnitude": [5.0, 6.0], "log_hypocentral_distance": [1.0, 2.0]})
    model = M.AttenuationBaseline().fit(frame, pd.Series([5.0, 4.0]))
    assert "MMI" in model.equation()


def test_baseline_respects_sample_weights():
    """A heavily weighted point should pull the fit towards itself."""
    frame = pd.DataFrame({
        "magnitude": [5.0, 5.0, 5.0],
        "log_hypocentral_distance": [1.0, 2.0, 3.0],
    })
    target = pd.Series([6.0, 5.0, 1.0])

    unweighted = M.AttenuationBaseline().fit(frame, target).predict(frame)
    weighted = M.AttenuationBaseline().fit(
        frame, target, sample_weight=[1, 1, 100]
    ).predict(frame)

    assert abs(weighted[2] - 1.0) < abs(unweighted[2] - 1.0)


# --- expected value ----------------------------------------------------------

def test_expected_mmi_is_the_probability_weighted_level():
    probabilities = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    result = M.expected_mmi(probabilities, [3, 4, 5])
    assert result[0] == pytest.approx(3.5)
    assert result[1] == pytest.approx(5.0)


def test_certain_prediction_returns_that_level():
    probabilities = np.array([[0.0, 1.0, 0.0]])
    assert M.expected_mmi(probabilities, [3, 4, 5])[0] == pytest.approx(4.0)


# --- metrics -----------------------------------------------------------------

def test_perfect_prediction_scores_perfectly():
    truth = np.array([3, 4, 5, 6])
    result = M.report_level_metrics(truth, truth)
    assert result["mae"] == 0
    assert result["within_1_mmi"] == 1.0
    assert result["bias"] == 0


def test_bias_shows_direction_not_just_size():
    truth = np.array([4, 4, 4])
    over = M.report_level_metrics(truth, truth + 1)
    under = M.report_level_metrics(truth, truth - 1)
    assert over["bias"] == pytest.approx(1.0)
    assert under["bias"] == pytest.approx(-1.0)
    assert over["mae"] == under["mae"]


def test_metrics_honour_sample_weights():
    truth = np.array([4, 4])
    predicted = np.array([4, 6])
    heavy_on_correct = M.report_level_metrics(truth, predicted, sample_weight=[100, 1])
    heavy_on_wrong = M.report_level_metrics(truth, predicted, sample_weight=[1, 100])
    assert heavy_on_correct["mae"] < heavy_on_wrong["mae"]


def test_cell_metrics_count_each_cell_once():
    """Report level metrics are dominated by dense cells; this view is not."""
    frame = pd.DataFrame({
        "public_id": ["e"] * 5,
        "cell_x": [0, 0, 0, 0, 1],
        "cell_y": [0, 0, 0, 0, 1],
        "predicted_mmi": [4.0] * 5,
        "mmi_mean": [4.0, 4.0, 4.0, 4.0, 6.0],
    })
    result = M.cell_level_metrics(frame)
    assert result["cells"] == 2
    # Two cells, one perfect and one out by 2, so mean absolute error is 1
    assert result["mae"] == pytest.approx(1.0)


def test_per_class_recall_exposes_ignored_levels():
    """A model that never predicts the rare high levels must score zero there."""
    truth = np.array([3, 3, 3, 8])
    predicted = np.array([3, 3, 3, 3])
    table = M.per_class_recall(truth, predicted, [3, 8]).set_index("mmi")

    assert table.loc[3, "recall"] == 1.0
    assert table.loc[8, "recall"] == 0.0


# --- weighting ---------------------------------------------------------------

def test_weight_schemes_change_influence_but_not_label_mix():
    frame = pd.DataFrame({
        "public_id": ["e", "e"], "cell_x": [0, 0], "cell_y": [0, 0],
        "mmi": [4, 5], "weight": [12.0, 9.0],
    })
    proportions = frame["weight"] / frame["weight"].sum()

    for scheme in M_SCHEMES:
        result = apply_weight_scheme(frame, scheme)
        assert np.allclose(result["weight"] / result["weight"].sum(), proportions)


M_SCHEMES = ("count", "sqrt", "equal")


def test_equal_scheme_gives_every_cell_one_unit():
    frame = pd.DataFrame({
        "public_id": ["a", "a", "b"], "cell_x": [0, 0, 1], "cell_y": [0, 0, 1],
        "mmi": [4, 5, 4], "weight": [90.0, 10.0, 3.0],
    })
    result = apply_weight_scheme(frame, "equal")
    per_cell = result.groupby(["public_id", "cell_x", "cell_y"])["weight"].sum()
    assert np.allclose(per_cell, 1.0)


def test_sqrt_scheme_sits_between_the_extremes():
    frame = pd.DataFrame({
        "public_id": ["a", "b"], "cell_x": [0, 1], "cell_y": [0, 1],
        "mmi": [4, 4], "weight": [100.0, 4.0],
    })
    ratios = {}
    for scheme in ("count", "sqrt", "equal"):
        weights = apply_weight_scheme(frame, scheme)["weight"]
        ratios[scheme] = weights.iloc[0] / weights.iloc[1]

    assert ratios["equal"] < ratios["sqrt"] < ratios["count"]


def test_unknown_weight_scheme_is_rejected():
    frame = pd.DataFrame({
        "public_id": ["a"], "cell_x": [0], "cell_y": [0], "mmi": [4], "weight": [1.0],
    })
    with pytest.raises(ValueError, match="unknown weight scheme"):
        apply_weight_scheme(frame, "inverse_vibes")
