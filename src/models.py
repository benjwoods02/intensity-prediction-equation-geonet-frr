"""Splitting, baseline, and evaluation for intensity prediction.

Splitting is by earthquake, never by row. Magnitude, depth and time of day
are properties of an event, not of a cell, so every cell of one earthquake
shares those values. Splitting rows at random would put cells from the same
event on both sides, letting a model recognise the event rather than learn how
shaking attenuates. With 95 events those features carry 95 independent
observations between them, not 24,241, and a row-wise split would hide that
completely.

The split is also stratified by magnitude. Large earthquakes are rare: there is
one above magnitude 7.5 in the whole catalogue. An unstratified split of 95
events could easily place every large event on one side, leaving a model that
has never seen strong shaking, or a test set that cannot measure it.

The baseline is the classical attenuation form, intensity as a linear function
of magnitude and the logarithm of distance. It is fitted here rather than taken
from a published equation with published coefficients, because the New Zealand
equations require rupture plane geometry that is not available from felt
reports alone. What it provides is the honest question: does a more complex
model beat the simplest form that respects the physics?
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

GROUP_COLUMN = "public_id"
CELL_COLUMNS = ["public_id", "cell_x", "cell_y"]


def event_table(frame, magnitude_column="magnitude", bin_width=0.5):
    """One row per earthquake, with the magnitude band used for stratification."""
    events = frame.groupby(GROUP_COLUMN, as_index=False)[magnitude_column].first()
    events["stratum_value"] = (events[magnitude_column] // bin_width) * bin_width

    # A stratum holding a single event cannot be split, and the sparse bands
    # are always the large magnitudes: there is one event above 7.5 in the
    # whole catalogue. Starting from the top, any band with fewer than two
    # events is folded down into the next one, repeatedly, until every band
    # can be split. This keeps the big earthquakes grouped together rather
    # than dropping them from stratification altogether.
    bands = sorted(events["stratum_value"].unique(), reverse=True)
    for index, band in enumerate(bands[:-1]):
        if (events["stratum_value"] == band).sum() < 2:
            events.loc[events["stratum_value"] == band, "stratum_value"] = bands[index + 1]

    events["stratum"] = events["stratum_value"].map(lambda x: f"M{x:.1f}+")
    return events.drop(columns="stratum_value")


def split_by_event(frame, test_size=0.2, random_state=7, verbose=True):
    """Split rows into train and test so that no earthquake appears in both."""
    events = event_table(frame)

    train_events, test_events = train_test_split(
        events[GROUP_COLUMN],
        test_size=test_size,
        random_state=random_state,
        stratify=events["stratum"],
    )

    train = frame[frame[GROUP_COLUMN].isin(train_events)].reset_index(drop=True)
    test = frame[frame[GROUP_COLUMN].isin(test_events)].reset_index(drop=True)

    if verbose:
        print(f"  train {len(train_events):>3} events, {len(train):>7,} rows")
        print(f"  test  {len(test_events):>3} events, {len(test):>7,} rows")
        overlap = set(train[GROUP_COLUMN]) & set(test[GROUP_COLUMN])
        print(f"  events appearing in both: {len(overlap)}")

    return train, test


def event_folds(frame, n_splits=5, random_state=7):
    """Yield train and test row masks for cross-validation grouped by earthquake."""
    events = event_table(frame)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in splitter.split(events, events["stratum"]):
        train_events = set(events.iloc[train_index][GROUP_COLUMN])
        test_events = set(events.iloc[test_index][GROUP_COLUMN])
        yield (frame[GROUP_COLUMN].isin(train_events).to_numpy(),
               frame[GROUP_COLUMN].isin(test_events).to_numpy())


class AttenuationBaseline(BaseEstimator, RegressorMixin):
    """Intensity as a linear function of magnitude and log distance.

    This is the shape almost every published intensity prediction equation
    takes, stripped to its essentials:

        MMI = c0 + c1 * magnitude + c2 * log10(hypocentral distance)

    Any more elaborate model has to beat this to justify itself. Fitting it on
    the same split as everything else keeps the comparison fair.
    """

    def __init__(self, magnitude_column="magnitude",
                 log_distance_column="log_hypocentral_distance"):
        self.magnitude_column = magnitude_column
        self.log_distance_column = log_distance_column

    def _design(self, X):
        return np.column_stack([X[self.magnitude_column], X[self.log_distance_column]])

    def fit(self, X, y, sample_weight=None):
        self.model_ = LinearRegression().fit(self._design(X), y, sample_weight=sample_weight)
        self.coefficients_ = {
            "intercept": self.model_.intercept_,
            "magnitude": self.model_.coef_[0],
            "log_distance": self.model_.coef_[1],
        }
        return self

    def predict(self, X):
        return self.model_.predict(self._design(X))

    def equation(self):
        c = self.coefficients_
        return (f"MMI = {c['intercept']:.3f} + {c['magnitude']:.3f} * M "
                f"{c['log_distance']:+.3f} * log10(R)")


def expected_mmi(probabilities, classes):
    """Collapse predicted class probabilities to a single expected intensity.

    A classifier returns how likely each intensity level is. Weighting the
    levels by those probabilities gives a continuous value, which is what
    shake maps and the attenuation check in the next phase need. It also means
    a confident wrong answer is penalised more than an uncertain one.
    """
    return probabilities @ np.asarray(classes, dtype="float64")


def report_level_metrics(y_true, y_predicted, sample_weight=None):
    """Accuracy of predicted intensity against individual reported levels."""
    error = np.asarray(y_predicted, dtype="float64") - np.asarray(y_true, dtype="float64")
    weights = np.ones(len(error)) if sample_weight is None else np.asarray(sample_weight)
    total = weights.sum()

    return {
        "mae": float((weights * np.abs(error)).sum() / total),
        "rmse": float(np.sqrt((weights * error ** 2).sum() / total)),
        "within_1_mmi": float((weights * (np.abs(error) <= 1)).sum() / total),
        "within_2_mmi": float((weights * (np.abs(error) <= 2)).sum() / total),
        "bias": float((weights * error).sum() / total),
    }


def cell_level_metrics(frame, predicted_column="predicted_mmi", observed_column="mmi_mean"):
    """Accuracy per square kilometre, with every cell counting once.

    Report level metrics are dominated by densely populated cells, because
    that is where the reports are. This view weights every cell equally, which
    is the fairer measure of whether the model describes the country rather
    than the cities.
    """
    cells = frame.groupby(CELL_COLUMNS, as_index=False).agg(
        predicted=(predicted_column, "first"),
        observed=(observed_column, "first"),
    )
    error = cells["predicted"] - cells["observed"]

    return {
        "cells": int(len(cells)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "within_1_mmi": float((error.abs() <= 1).mean()),
        "within_2_mmi": float((error.abs() <= 2).mean()),
        "bias": float(error.mean()),
    }


def per_class_recall(y_true, y_predicted_class, classes):
    """How often each intensity level is recovered.

    Aggregate accuracy hides the levels that matter. MMI 7 and 8 together are
    about 1.3% of reports, so a model that never predicts above 6 can still
    look good overall while being useless for damaging shaking.
    """
    y_true = np.asarray(y_true)
    y_predicted_class = np.asarray(y_predicted_class)

    rows = []
    for level in classes:
        actual = y_true == level
        rows.append({
            "mmi": level,
            "reports": int(actual.sum()),
            "share_%": round(100 * actual.mean(), 2),
            "recall": round(float((y_predicted_class[actual] == level).mean()), 3) if actual.any() else np.nan,
        })

    return pd.DataFrame(rows)


def ranked_probability_score(probabilities, y_true, classes, sample_weight=None):
    """Ranked Probability Score for ordinal forecasts. Lower is better.

    This is the metric the "within 1 MMI" score should probably have been all
    along. It compares the predicted cumulative distribution with the observed
    one, squared and summed across levels:

        RPS = mean over samples of  sum_k (F_pred(k) - F_obs(k))^2 / (K - 1)

    Three properties make it the right fit here.

    It respects the ordering. Predicting MMI 7 when the answer is 8 costs far
    less than predicting 3, which plain log loss or a Brier score would treat
    as equally wrong.

    It is proper, meaning the best score comes from reporting your honest
    belief. There is no equivalent of the integer-snapping trick that lets a
    constant prediction win on within-1.

    It uses the whole predicted distribution rather than collapsing it, so a
    model that is uncertain in the right way scores better than one that is
    confidently wrong.

    It is also the standard score for ordinal and categorical forecasts in
    meteorology and seismology, so it is not an invention of this project.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)

    predicted_cdf = probabilities.cumsum(axis=1)
    observed_cdf = (classes[None, :] >= y_true[:, None]).astype("float64")

    per_sample = ((predicted_cdf - observed_cdf) ** 2).sum(axis=1) / (len(classes) - 1)

    if sample_weight is None:
        return float(per_sample.mean())

    weights = np.asarray(sample_weight, dtype="float64")
    return float((weights * per_sample).sum() / weights.sum())


def balanced_class_weights(y, classes):
    """Weight each intensity level inversely to how often it appears.

    MMI 7 and 8 are about 1.3% of reports between them, so an unweighted fit
    can ignore them entirely and still score well. Balancing makes a rare high
    intensity worth as much in the loss as a common low one.

    Returned as a mapping so it can be folded into the existing sample weights
    rather than relying on a class_weight argument, which regressors and
    several classifiers do not accept.
    """
    y = np.asarray(y)
    counts = {level: max((y == level).sum(), 1) for level in classes}
    total = sum(counts.values())
    return {level: total / (len(classes) * count) for level, count in counts.items()}


def apply_class_weights(frame, classes, target_column="mmi", weight_column="weight"):
    """Fold balanced class weights into the existing per-row sample weights.

    The cell weighting decides how much a location counts. The class weighting
    decides how much an intensity level counts. Multiplying them keeps both.
    """
    weights = balanced_class_weights(frame[target_column], classes)
    scaled = frame[weight_column] * frame[target_column].map(weights)
    return frame.assign(**{weight_column: scaled})
