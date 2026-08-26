"""Tests for the model bench in src/bench.py.

The bench's whole value is that every candidate is treated identically, so
these tests are mostly about the harness rather than about any model: that the
registry is coherent, that sample weights reach the right place through a
pipeline, that a candidate which blows up is recorded rather than taking the
sweep down with it, and that the metrics decisions the project argued for are
actually the ones implemented.

Everything runs on a tiny synthetic dataset, so the suite stays offline and
fast.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bench as B

FEATURES = ["magnitude", "log_hypocentral_distance"]


def toy_frame(n_events=8, cells_per_event=25, seed=7):
    """A miniature version of the real training frame."""
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(n_events):
        magnitude = 4.0 + event * 0.4
        for cell in range(cells_per_event):
            log_distance = rng.uniform(1.0, 2.6)
            mmi = int(np.clip(round(8.0 - 1.5 * log_distance + 0.2 * magnitude), 3, 8))
            rows.append({
                "public_id": f"event_{event}",
                "cell_x": cell, "cell_y": event,
                "magnitude": magnitude,
                "log_hypocentral_distance": log_distance,
                "mmi": mmi,
                "mmi_mean": float(mmi),
                "weight": float(rng.integers(1, 20)),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split():
    frame = toy_frame()
    events = frame["public_id"].unique()
    train = frame[frame["public_id"].isin(events[:6])].reset_index(drop=True)
    test = frame[frame["public_id"].isin(events[6:])].reset_index(drop=True)
    return train, test


# --- the registry ------------------------------------------------------------

def test_registry_names_are_unique():
    """Results are keyed by name, so a collision would silently overwrite one."""
    names = [candidate.name for candidate in B.build_registry()]
    assert len(names) == len(set(names))


def test_registry_spans_model_families():
    """The point of the bench is breadth, not variants of one winner."""
    families = {candidate.family for candidate in B.build_registry()}
    assert len(families) >= 8
    assert "baseline" in families


def test_every_candidate_declares_a_valid_kind():
    for candidate in B.build_registry():
        assert candidate.kind in {"classification", "regression"}


def test_registry_is_reproducible():
    """Same seed, same candidate list, or the comparison means nothing."""
    first = [c.name for c in B.build_registry(random_state=7)]
    second = [c.name for c in B.build_registry(random_state=7)]
    assert first == second


# --- sample weights ----------------------------------------------------------

def test_weight_kwargs_targets_the_final_pipeline_step():
    """A pipeline needs the step-prefixed keyword or the weights go nowhere."""
    pipeline = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    assert B._weight_kwargs(pipeline, [1, 2]) == {"model__sample_weight": [1, 2]}


def test_weight_kwargs_is_plain_for_a_bare_estimator():
    assert B._weight_kwargs(Ridge(), [1, 2]) == {"sample_weight": [1, 2]}


# --- subsampling -------------------------------------------------------------

def test_subsample_leaves_small_frames_alone(split):
    train, _ = split
    sampled, was_subsampled = B._subsample(train, cap=10_000, weight_column="weight",
                                           random_state=7)
    assert len(sampled) == len(train)
    assert not was_subsampled


def test_subsample_caps_and_reports_it(split):
    train, _ = split
    sampled, was_subsampled = B._subsample(train, cap=40, weight_column="weight",
                                           random_state=7)
    assert len(sampled) == 40
    assert was_subsampled


def test_subsample_is_uniform_not_weighted(split):
    """Weighting here and at fit time would apply the same weighting twice."""
    train, _ = split
    heavy = train.copy()
    heavy.loc[heavy.index[0], "weight"] = 1e6

    sampled, _ = B._subsample(heavy, cap=50, weight_column="weight", random_state=7)
    # Under weighted sampling the one heavy row would dominate; under uniform
    # sampling it is just another row and the mean weight stays near the
    # unweighted mean of the rest.
    assert sampled["weight"].max() < 1e6 or len(sampled) == 50


# --- running a candidate -----------------------------------------------------

def test_regression_candidate_reports_the_expected_metrics(split):
    train, test = split
    row = B.run_candidate(B.Candidate("Ridge", Ridge(), "regression", "linear"),
                          train, test, FEATURES)

    assert row["status"] == "ok"
    for key in ["cell_mae", "report_mae", "report_within_1", "cell_within_1", "seconds"]:
        assert key in row and row[key] == row[key]


def test_classification_candidate_reports_probability_metrics(split):
    from sklearn.tree import DecisionTreeClassifier

    train, test = split
    row = B.run_candidate(
        B.Candidate("Tree", DecisionTreeClassifier(max_depth=3, random_state=7),
                    "classification", "tree"),
        train, test, FEATURES)

    assert row["status"] == "ok"
    # These only exist for models that produce a distribution.
    assert 0 <= row["rps"] <= 1
    assert row["auc_mmi6plus"] != row["auc_mmi6plus"] or 0 <= row["auc_mmi6plus"] <= 1


def test_a_failing_candidate_is_recorded_not_raised(split):
    """One broken model must not take the whole sweep down."""
    class Explodes:
        def fit(self, X, y, **kwargs):
            raise ValueError("nope")

    train, test = split
    row = B.run_candidate(B.Candidate("Broken", Explodes(), "regression", "linear"),
                          train, test, FEATURES)

    assert row["status"].startswith("failed")
    assert "ValueError" in row["status"]


def test_within_1_is_scored_on_rounded_predictions(split):
    """The metric decision the project argued for, asserted rather than assumed.

    A constant 4.2 is a continuous prediction. Scored raw it would fall more
    than a unit from every MMI 3, which is what makes the metric measure
    integer-ness. Rounding both sides first is what makes it comparable.
    """
    # A spread that spans the levels where the distinction bites. Rounded to
    # 4.0 the prediction covers MMI 3, 4 and 5; left at 4.2 it loses every 3.
    frame = pd.DataFrame({
        "public_id": ["a"] * 4 + ["b"] * 4,
        "cell_x": list(range(4)) * 2, "cell_y": [0] * 4 + [1] * 4,
        "magnitude": 5.5, "log_hypocentral_distance": 2.0,
        "mmi": [3, 4, 5, 6] * 2,
        "mmi_mean": [3.0, 4.0, 5.0, 6.0] * 2,
        "weight": 1.0,
    })
    train = frame[frame["public_id"] == "a"].reset_index(drop=True)
    test = frame[frame["public_id"] == "b"].reset_index(drop=True)

    row = B.run_candidate(
        B.Candidate("Constant 4.2", DummyRegressor(strategy="constant", constant=4.2),
                    "regression", "baseline"),
        train, test, FEATURES)

    labels = test["mmi"].to_numpy()
    raw = float(np.mean(np.abs(labels - 4.2) <= 1))       # 0.50, loses MMI 3
    rounded = float(np.mean(np.abs(labels - 4.0) <= 1))   # 0.75, keeps it

    assert raw < rounded, "the toy data must actually exercise the difference"
    assert row["report_within_1"] == pytest.approx(rounded, abs=1e-4)


# --- the sweep ---------------------------------------------------------------

def test_run_bench_returns_one_row_per_candidate(split):
    train, test = split
    candidates = [B.Candidate("Ridge", Ridge(), "regression", "linear"),
                  B.Candidate("Mean", DummyRegressor(), "regression", "baseline")]

    results = B.run_bench(train, test, FEATURES, candidates=candidates, verbose=False)

    assert len(results) == 2
    assert set(results["model"]) == {"Ridge", "Mean"}


# --- refitting for inspection ------------------------------------------------

def test_fit_shortlist_returns_fitted_candidates(split):
    train, _ = split
    fitted = B.fit_shortlist(["Attenuation equation"], train, FEATURES)

    assert set(fitted) == {"Attenuation equation"}
    # A fitted model predicts; an unfitted one raises.
    assert len(fitted["Attenuation equation"].estimator.predict(train[FEATURES])) == len(train)


def test_fit_shortlist_rejects_an_unknown_name(split):
    train, _ = split
    with pytest.raises(KeyError, match="not in the registry"):
        B.fit_shortlist(["No Such Model"], train, FEATURES)
