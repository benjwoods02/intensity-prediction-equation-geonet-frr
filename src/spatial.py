"""Check whether a model's predictions are physically possible.

Predictive accuracy is not sufficient here. Shaking must weaken as you move
away from an earthquake: that is not a preference, it is what attenuation
means. A model can score well on held out data and still produce a map where
intensity rises with distance, or wanders up and down, because nothing in the
loss function tells it that distance has a direction.

This is the check the original project made by eye, generating shake maps for
each candidate and looking at them. Two problems with that. It does not scale
past a handful of models, and it cannot be repeated by anyone else. Here the
same judgement is made numerically, so every candidate faces the same test and
the result is reproducible.

Two tools:

  radial_profile   Holds an earthquake fixed and sweeps distance outwards,
                   averaging over compass directions so a single unlucky
                   bearing cannot decide the answer. This isolates the
                   distance response from everything else.

  shake_map        Predicts across a grid covering the country, for the
                   picture a person actually reads.

A model is judged on whether its radial profile falls, and by how cleanly.
Failing this is disqualifying regardless of error metrics, because a map that
shows shaking increasing with distance is worse than useless: it would send
people to the wrong places.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clean import from_nztm, to_nztm
from features import encode_circular
from models import expected_mmi

MMI_CLASSES = np.arange(3, 9)

# Held constant while distance is swept, so only distance moves. Values are
# typical rather than extreme: a median site, early afternoon on a weekday.
REFERENCE_CONDITIONS = {
    "vs30": 400.0,
    "local_hour": 14.0,
    "is_weekend": 0,
}


def predict_intensity(model, features, kind="classification"):
    """Predict a continuous intensity from either a classifier or a regressor."""
    if kind != "classification":
        return np.asarray(model.predict(features), dtype="float64")

    probabilities = model.predict_proba(features)
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    return expected_mmi(probabilities, classes)


def _feature_frame(distances_km, magnitude, depth_km, azimuths, conditions=None):
    """Build model input for a set of distances and bearings."""
    conditions = {**REFERENCE_CONDITIONS, **(conditions or {})}

    distances, bearings = np.meshgrid(np.asarray(distances_km, dtype="float64"),
                                      np.asarray(azimuths, dtype="float64"))
    distances = distances.ravel()
    bearings = bearings.ravel()

    hypocentral = np.sqrt(distances ** 2 + depth_km ** 2)
    azimuth_sin, azimuth_cos = encode_circular(bearings, period=360)
    hour_sin, hour_cos = encode_circular(np.full_like(distances, conditions["local_hour"]), period=24)

    return pd.DataFrame({
        "magnitude": magnitude,
        "depth_km": depth_km,
        "log_hypocentral_distance": np.log10(hypocentral),
        "azimuth_sin": azimuth_sin,
        "azimuth_cos": azimuth_cos,
        "local_hour_sin": hour_sin,
        "local_hour_cos": hour_cos,
        "is_weekend": conditions["is_weekend"],
        "vs30": conditions["vs30"],
        "epicentral_distance_km": distances,
    })


def radial_profile(model, feature_columns, magnitude=6.0, depth_km=15.0,
                   distances_km=None, n_azimuths=8, kind="classification",
                   conditions=None):
    """Predicted intensity as a function of distance, averaged over bearings.

    Averaging over compass directions matters. A model given azimuth features
    can produce a profile that falls in one direction and rises in another, and
    testing a single bearing would report whichever it happened to pick.
    """
    if distances_km is None:
        distances_km = np.geomspace(5, 400, 40)

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    frame = _feature_frame(distances_km, magnitude, depth_km, azimuths, conditions)

    frame["predicted_mmi"] = predict_intensity(model, frame[feature_columns], kind=kind)

    return (frame.groupby("epicentral_distance_km", as_index=False)["predicted_mmi"]
            .mean()
            .sort_values("epicentral_distance_km")
            .reset_index(drop=True))


def attenuation_check(profile):
    """Score how well a radial profile behaves like real attenuation.

    Three numbers, because they fail in different ways:

      spearman            Rank correlation between distance and intensity.
                          Should be close to -1. A positive value means the
                          model has intensity growing with distance.

      fraction_decreasing Share of adjacent steps that go down. A model can
                          trend downwards overall while oscillating, which
                          produces a blotchy map.

      total_drop          How much intensity falls from the nearest distance
                          to the furthest. A model can be perfectly monotonic
                          and still useless if it barely varies.
    """
    intensity = profile["predicted_mmi"].to_numpy()
    distance = profile["epicentral_distance_km"].to_numpy()

    if len(intensity) < 3 or np.allclose(intensity, intensity[0]):
        return {"spearman": np.nan, "fraction_decreasing": np.nan,
                "total_drop": 0.0, "monotonic": False}

    steps = np.diff(intensity)
    correlation = spearmanr(distance, intensity).statistic

    return {
        "spearman": float(correlation),
        "fraction_decreasing": float((steps <= 0).mean()),
        "total_drop": float(intensity[0] - intensity[-1]),
        "monotonic": bool((steps <= 1e-9).all()),
    }


def physical_plausibility(model, feature_columns, kind="classification",
                          magnitudes=(4.5, 5.5, 6.5, 7.5), depth_km=15.0):
    """Run the attenuation check across several magnitudes.

    A model can behave for a moderate earthquake and misbehave for a large one,
    usually because it saw few large events in training. Testing one magnitude
    would hide that.
    """
    rows = []
    for magnitude in magnitudes:
        profile = radial_profile(model, feature_columns, magnitude=magnitude,
                                 depth_km=depth_km, kind=kind)
        rows.append({"magnitude": magnitude, **attenuation_check(profile)})

    table = pd.DataFrame(rows)

    return {
        "worst_spearman": float(table["spearman"].max()),
        "mean_fraction_decreasing": float(table["fraction_decreasing"].mean()),
        "mean_total_drop": float(table["total_drop"].mean()),
        "monotonic_at_all_magnitudes": bool(table["monotonic"].all()),
        "by_magnitude": table,
    }


def nz_grid(spacing_km=10):
    """A grid of points covering mainland New Zealand, for mapping."""
    easting_min, easting_max = 1.05e6, 2.10e6
    northing_min, northing_max = 4.75e6, 6.20e6
    step = spacing_km * 1000

    eastings = np.arange(easting_min, easting_max, step)
    northings = np.arange(northing_min, northing_max, step)
    grid_easting, grid_northing = np.meshgrid(eastings, northings)

    longitude, latitude = from_nztm(grid_easting.ravel(), grid_northing.ravel())

    return pd.DataFrame({
        "easting": grid_easting.ravel(),
        "northing": grid_northing.ravel(),
        "longitude": longitude,
        "latitude": latitude,
    })


def shake_map(model, feature_columns, event, grid=None, kind="classification",
              conditions=None):
    """Predict intensity across the country for one earthquake.

    event needs magnitude, depth_km, longitude and latitude. Site conditions
    are held at reference values, so the map shows the model's view of source
    and distance rather than local ground effects.
    """
    from features import haversine_km, initial_bearing_degrees

    grid = nz_grid() if grid is None else grid
    conditions = {**REFERENCE_CONDITIONS, **(conditions or {})}

    distance = haversine_km(event["longitude"], event["latitude"],
                            grid["longitude"], grid["latitude"])
    bearing = initial_bearing_degrees(event["longitude"], event["latitude"],
                                      grid["longitude"], grid["latitude"])

    hypocentral = np.sqrt(distance ** 2 + event["depth_km"] ** 2)
    azimuth_sin, azimuth_cos = encode_circular(bearing, period=360)
    hour_sin, hour_cos = encode_circular(np.full(len(grid), conditions["local_hour"]), period=24)

    frame = grid.assign(
        magnitude=event["magnitude"],
        depth_km=event["depth_km"],
        epicentral_distance_km=distance,
        log_hypocentral_distance=np.log10(hypocentral),
        azimuth_sin=azimuth_sin,
        azimuth_cos=azimuth_cos,
        local_hour_sin=hour_sin,
        local_hour_cos=hour_cos,
        is_weekend=conditions["is_weekend"],
        vs30=conditions["vs30"],
    )

    frame["predicted_mmi"] = predict_intensity(model, frame[feature_columns], kind=kind)
    return frame


# Thresholds for accepting a model as physically plausible. Chosen after
# looking at how the candidates actually behave, and deliberately not set at
# strict monotonicity.
#
# Requiring every single step to decrease sounds right and is the wrong test.
# Tree ensembles predict piecewise constant surfaces, so their profiles wobble
# by thousandths of an MMI unit between neighbouring distances while falling
# cleanly overall. That test passes only shallow decision trees, whose profiles
# are monotonic because they are nearly flat: they drop about 0.4 MMI across
# the entire country and would produce a shake map that says almost nothing.
#
# So the test is on the shape of the decay rather than its every step: a strong
# overall downward trend, few reversals, and enough range to be worth mapping.
MINIMUM_SPEARMAN = -0.95
MINIMUM_FRACTION_DECREASING = 0.90
MINIMUM_TOTAL_DROP = 0.5


def plausibility_verdict(summary):
    """Decide whether a model's attenuation behaviour is acceptable.

    Returns the verdict and the reasons, so a rejection can be explained
    rather than asserted.
    """
    reasons = []

    if not summary["worst_spearman"] <= MINIMUM_SPEARMAN:
        reasons.append(
            f"weak or inconsistent decay (worst Spearman {summary['worst_spearman']:.3f}, "
            f"needs {MINIMUM_SPEARMAN})")

    if summary["mean_fraction_decreasing"] < MINIMUM_FRACTION_DECREASING:
        reasons.append(
            f"too many reversals ({100 * (1 - summary['mean_fraction_decreasing']):.0f}% of steps "
            f"increase with distance)")

    if summary["mean_total_drop"] < MINIMUM_TOTAL_DROP:
        reasons.append(
            f"barely varies across the country (drops only "
            f"{summary['mean_total_drop']:.2f} MMI)")

    return {"passes": not reasons, "reasons": reasons}
