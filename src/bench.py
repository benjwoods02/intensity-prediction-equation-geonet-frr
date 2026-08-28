"""A repeatable bench for comparing many models on identical terms.

Every candidate sees the same split, the same features and the same sample
weights, and is scored with the same metrics.

Both classification and regression candidates are included. The project frames
the problem as ordinal classification, so a classifier's predicted
probabilities are collapsed to an expected intensity, which puts it on the
same continuous scale as a regressor and the baseline. Regressors are kept in
the comparison because the classical attenuation equation is one.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNet, HuberRegressor, Lasso,
                                  LinearRegression, LogisticRegression, Ridge,
                                  SGDClassifier, SGDRegressor)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVR, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from models import (AttenuationBaseline, apply_class_weights, cell_level_metrics,
                    expected_mmi, high_intensity_discrimination, per_class_recall,
                    ranked_probability_score,
                    report_level_metrics)

MMI_CLASSES = np.arange(3, 9)


@dataclass
class Candidate:
    """One model on the bench, plus what the harness needs to know to run it."""

    name: str
    estimator: Any
    kind: str
    family: str
    supports_sample_weight: bool = True
    max_train_rows: Optional[int] = None
    notes: str = ""


def _scaled(estimator):
    """Wrap in standardisation, for models that care about feature scale."""
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _polynomial(estimator, degree=2):
    return Pipeline([
        ("scale", StandardScaler()),
        ("expand", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", estimator),
    ])


def build_registry(random_state=7):
    """The candidate list.
    """
    seed = random_state
    candidates = [
        # --- reference points -------------------------------------------
        Candidate("Dummy (most frequent)", DummyClassifier(strategy="most_frequent"),
                  "classification", "baseline",
                  notes="predicts the commonest intensity for everything"),
        Candidate("Dummy (mean)", DummyRegressor(strategy="mean"),
                  "regression", "baseline",
                  notes="predicts the average intensity for everything"),
        Candidate("Attenuation equation", AttenuationBaseline(), "regression", "baseline",
                  notes="the classical form: intensity from magnitude and log distance"),

        # --- linear -----------------------------------------------------
        Candidate("Logistic Regression", _scaled(LogisticRegression(max_iter=2000, random_state=seed)),
                  "classification", "linear"),
        Candidate("Logistic Regression (L1)",
                  _scaled(LogisticRegression(penalty="l1", solver="saga", max_iter=2000, random_state=seed)),
                  "classification", "linear"),
        Candidate("Logistic Regression (poly 2)",
                  _polynomial(LogisticRegression(max_iter=2000, random_state=seed)),
                  "classification", "linear"),
        Candidate("SGD Classifier (log loss)",
                  _scaled(SGDClassifier(loss="log_loss", max_iter=2000, random_state=seed)),
                  "classification", "linear"),
        Candidate("Linear Regression", _scaled(LinearRegression()), "regression", "linear"),
        Candidate("Ridge", _scaled(Ridge(random_state=seed)), "regression", "linear"),
        Candidate("Ridge (poly 2)", _polynomial(Ridge(random_state=seed)), "regression", "linear"),
        Candidate("Ridge (poly 3)", _polynomial(Ridge(random_state=seed), degree=3),
                  "regression", "linear"),
        Candidate("Lasso", _scaled(Lasso(random_state=seed)), "regression", "linear"),
        Candidate("Elastic Net", _scaled(ElasticNet(random_state=seed)), "regression", "linear"),
        Candidate("Bayesian Ridge", _scaled(BayesianRidge()), "regression", "linear"),
        Candidate("Huber Regressor", _scaled(HuberRegressor(max_iter=500)), "regression", "linear",
                  notes="robust to outlying reports"),
        Candidate("SGD Regressor", _scaled(SGDRegressor(max_iter=2000, random_state=seed)),
                  "regression", "linear"),

        # --- discriminant and Bayes -------------------------------------
        Candidate("Linear Discriminant", _scaled(LinearDiscriminantAnalysis()),
                  "classification", "discriminant"),
        Candidate("Quadratic Discriminant", _scaled(QuadraticDiscriminantAnalysis()),
                  "classification", "discriminant", supports_sample_weight=False),
        Candidate("Gaussian Naive Bayes", _scaled(GaussianNB()), "classification", "bayes"),

        # --- distance based ---------------------------------------------
        Candidate("KNN (k=5)", _scaled(KNeighborsClassifier(n_neighbors=5)),
                  "classification", "neighbours", supports_sample_weight=False),
        Candidate("KNN (k=25)", _scaled(KNeighborsClassifier(n_neighbors=25)),
                  "classification", "neighbours", supports_sample_weight=False),
        Candidate("KNN (k=100)", _scaled(KNeighborsClassifier(n_neighbors=100)),
                  "classification", "neighbours", supports_sample_weight=False),
        Candidate("KNN Regressor (k=25)", _scaled(KNeighborsRegressor(n_neighbors=25)),
                  "regression", "neighbours", supports_sample_weight=False),

        # --- trees ------------------------------------------------------
        Candidate("Decision Tree (depth 5)",
                  DecisionTreeClassifier(max_depth=5, random_state=seed), "classification", "tree"),
        Candidate("Decision Tree (depth 12)",
                  DecisionTreeClassifier(max_depth=12, random_state=seed), "classification", "tree"),
        Candidate("Decision Tree Regressor (depth 8)",
                  DecisionTreeRegressor(max_depth=8, random_state=seed), "regression", "tree"),
        Candidate("Random Forest", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                  "classification", "ensemble"),
        Candidate("Random Forest (depth 10)",
                  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1),
                  "classification", "ensemble"),
        Candidate("Extra Trees", ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                  "classification", "ensemble"),
        Candidate("Random Forest Regressor",
                  RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
                  "regression", "ensemble"),
        Candidate("Extra Trees Regressor",
                  ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
                  "regression", "ensemble"),
        Candidate("Bagging", BaggingClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
                  "classification", "ensemble"),

        # --- boosting ---------------------------------------------------
        Candidate("Gradient Boosting", GradientBoostingClassifier(random_state=seed),
                  "classification", "boosting", max_train_rows=20000,
                  notes="slow at full size"),
        Candidate("Hist Gradient Boosting", HistGradientBoostingClassifier(random_state=seed),
                  "classification", "boosting"),
        Candidate("Hist Gradient Boosting (depth 4)",
                  HistGradientBoostingClassifier(max_depth=4, random_state=seed),
                  "classification", "boosting"),
        Candidate("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=seed),
                  "classification", "boosting"),
        Candidate("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=seed),
                  "regression", "boosting"),
        Candidate("Hist Gradient Boosting Regressor",
                  HistGradientBoostingRegressor(random_state=seed), "regression", "boosting"),

        # --- neural -----------------------------------------------------
        Candidate("MLP (64)", _scaled(MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                                    random_state=seed)),
                  "classification", "neural", supports_sample_weight=False),
        Candidate("MLP (128, 64)", _scaled(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                                         random_state=seed)),
                  "classification", "neural", supports_sample_weight=False),
        Candidate("MLP Regressor (64)", _scaled(MLPRegressor(hidden_layer_sizes=(64,), max_iter=300,
                                                             random_state=seed)),
                  "regression", "neural", supports_sample_weight=False),

        # --- kernel -----------------------------------------------------
        Candidate("SVC (rbf)", _scaled(SVC(probability=True, random_state=seed)),
                  "classification", "kernel", max_train_rows=6000,
                  notes="quadratic in sample count, so fitted on a subsample"),
        Candidate("SVR (rbf)", _scaled(SVR()), "regression", "kernel",
                  max_train_rows=6000, supports_sample_weight=True,
                  notes="quadratic in sample count, so fitted on a subsample"),
        Candidate("Linear SVR", _scaled(LinearSVR(max_iter=5000, random_state=seed)),
                  "regression", "kernel"),
    ]
    return candidates


def _subsample(train, cap, weight_column, random_state):
    """Take a plain random sample of the training rows, for models that cannot scale.
    """
    if cap is None or len(train) <= cap:
        return train, False
    return train.sample(cap, random_state=random_state), True


def run_candidate(candidate, train, test, feature_columns, target_column="mmi",
                  weight_column="weight", random_state=7):
    """Fit and score one candidate, returning a row of results or a failure note."""
    fit_rows, was_subsampled = _subsample(train, candidate.max_train_rows, weight_column, random_state)

    X_train = fit_rows[feature_columns]
    y_train = fit_rows[target_column]
    X_test = test[feature_columns]

    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if candidate.supports_sample_weight:
                try:
                    candidate.estimator.fit(X_train, y_train,
                                            **_weight_kwargs(candidate.estimator, fit_rows[weight_column]))
                except TypeError:
                    candidate.estimator.fit(X_train, y_train)
            else:
                candidate.estimator.fit(X_train, y_train)

            rps = auc6 = auc7 = np.nan
            if candidate.kind == "classification":
                probabilities = candidate.estimator.predict_proba(X_test)
                classes = candidate.estimator.classes_ if hasattr(candidate.estimator, "classes_") \
                    else candidate.estimator[-1].classes_
                predicted = expected_mmi(probabilities, classes)
                predicted_class = np.asarray(classes)[probabilities.argmax(axis=1)]

                # A classifier that saw only some levels in training returns a
                # narrower probability matrix, so pad it back to the full scale
                # before scoring against the fixed set of intensities.
                full = np.zeros((len(probabilities), len(MMI_CLASSES)))
                for position, level in enumerate(classes):
                    full[:, list(MMI_CLASSES).index(level)] = probabilities[:, position]
                rps = ranked_probability_score(full, test[target_column], MMI_CLASSES,
                                               test[weight_column])
                auc6 = high_intensity_discrimination(full, test[target_column], MMI_CLASSES,
                                                     6, test[weight_column])
                auc7 = high_intensity_discrimination(full, test[target_column], MMI_CLASSES,
                                                     7, test[weight_column])
            else:
                predicted = candidate.estimator.predict(X_test)
                predicted_class = np.clip(np.rint(predicted), 3, 8).astype(int)

    except Exception as error:
        return {"model": candidate.name, "family": candidate.family, "kind": candidate.kind,
                "status": f"failed: {type(error).__name__}", "seconds": round(time.perf_counter() - started, 1)}

    report = report_level_metrics(test[target_column], predicted, test[weight_column])
    # within-1 is scored on the rounded prediction. Against a continuous value
    # it rewards landing exactly on an integer rather than being close, which
    # is enough to let a constant outscore a strictly better prediction.
    rounded = report_level_metrics(test[target_column], predicted_class, test[weight_column])
    cells = cell_level_metrics(test.assign(predicted_mmi=predicted))
    recall = per_class_recall(test[target_column], predicted_class, MMI_CLASSES)
    high = recall[recall["mmi"] >= 7]["recall"]

    return {
        "model": candidate.name,
        "family": candidate.family,
        "kind": candidate.kind,
        "status": "subsampled" if was_subsampled else "ok",
        "train_rows": len(fit_rows),
        "rps": round(rps, 4) if rps == rps else np.nan,
        "auc_mmi6plus": round(auc6, 4) if auc6 == auc6 else np.nan,
        "auc_mmi7plus": round(auc7, 4) if auc7 == auc7 else np.nan,
        "report_within_1": round(rounded["within_1_mmi"], 4),
        "report_mae": round(report["mae"], 4),
        "report_bias": round(report["bias"], 4),
        "cell_within_1": round(cells["within_1_mmi"], 4),
        "cell_mae": round(cells["mae"], 4),
        # nanmean over an all-NaN slice warns rather than returning NaN quietly,
        # and every level can be absent from a small test set.
        "recall_mmi7plus": round(float(np.nanmean(high)), 4) if high.notna().any() else np.nan,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _weight_kwargs(estimator, weights):
    """Sample weights go by a different keyword inside a pipeline."""
    if isinstance(estimator, Pipeline):
        return {f"{estimator.steps[-1][0]}__sample_weight": weights}
    return {"sample_weight": weights}


def fit_shortlist(names, train, feature_columns, target_column="mmi",
                  weight_column="weight", random_state=7):
    """Refit named candidates on the whole training set, for inspection.
    """
    wanted = set(names)
    fitted = {}

    for candidate in build_registry(random_state):
        if candidate.name not in wanted:
            continue
        try:
            candidate.estimator.fit(
                train[feature_columns], train[target_column],
                **_weight_kwargs(candidate.estimator, train[weight_column]))
        except TypeError:
            # Not every estimator accepts sample weights.
            candidate.estimator.fit(train[feature_columns], train[target_column])
        fitted[candidate.name] = candidate

    missing = wanted - set(fitted)
    if missing:
        raise KeyError(f"not in the registry: {sorted(missing)}")

    return fitted


def run_bench(train, test, feature_columns, candidates=None, random_state=7,
              class_weighted=False, verbose=True):
    """Run every candidate and return one comparison table.

    class_weighted folds balanced class weights into the training sample
    weights, so a rare high intensity counts as much in the loss as a common
    low one. Without it no model predicts MMI 7 or 8 at all.
    """
    candidates = candidates if candidates is not None else build_registry(random_state)
    if class_weighted:
        train = apply_class_weights(train, MMI_CLASSES)
    results = []

    for index, candidate in enumerate(candidates, start=1):
        if verbose:
            print(f"[{index:>2}/{len(candidates)}] {candidate.name:<36}", end="", flush=True)

        row = run_candidate(candidate, train, test, feature_columns, random_state=random_state)
        results.append(row)

        if verbose:
            if row["status"].startswith("failed"):
                print(f" {row['status']}")
            else:
                print(f" cell MAE {row['cell_mae']:.3f}  report MAE {row['report_mae']:.3f}  ({row['seconds']:.0f}s)")

    return pd.DataFrame(results)
