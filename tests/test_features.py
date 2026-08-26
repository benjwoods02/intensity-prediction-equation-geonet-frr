"""Tests for the feature derivation in src/features.py.

Distances and bearings are checked against independently known values rather
than against the implementation's own output, so a wrong formula fails rather
than being enshrined.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clean import from_nztm
from features import (MODEL_FEATURES, add_local_time_features, add_vs30,
                      build_features, encode_circular, haversine_km,
                      initial_bearing_degrees, load_vs30_grid, model_features)

CHRISTCHURCH = (172.6376, -43.5309)
WELLINGTON = (174.7762, -41.2785)
AUCKLAND = (174.7625, -36.8485)


# --- distance ----------------------------------------------------------------

@pytest.mark.parametrize("start, end, expected_km", [
    (CHRISTCHURCH, WELLINGTON, 305),
    (AUCKLAND, WELLINGTON, 494),
    (AUCKLAND, CHRISTCHURCH, 764),
])
def test_distance_matches_published_values(start, end, expected_km):
    distance = haversine_km(start[0], start[1], end[0], end[1])
    assert float(distance) == pytest.approx(expected_km, abs=5)


def test_distance_to_self_is_zero():
    assert float(haversine_km(*CHRISTCHURCH, *CHRISTCHURCH)) == pytest.approx(0, abs=1e-9)


def test_distance_is_symmetric():
    there = haversine_km(*CHRISTCHURCH, *WELLINGTON)
    back = haversine_km(*WELLINGTON, *CHRISTCHURCH)
    assert float(there) == pytest.approx(float(back))


# --- bearing -----------------------------------------------------------------

@pytest.mark.parametrize("delta_lon, delta_lat, expected", [
    (0.0, 1.0, 0.0),      # north
    (1.0, 0.0, 90.0),     # east
    (0.0, -1.0, 180.0),   # south
    (-1.0, 0.0, 270.0),   # west
])
def test_bearing_points_the_right_way(delta_lon, delta_lat, expected):
    bearing = initial_bearing_degrees(174.0, -41.0, 174.0 + delta_lon, -41.0 + delta_lat)
    assert float(bearing) == pytest.approx(expected, abs=1.0)


def test_bearing_stays_within_a_full_turn():
    bearings = initial_bearing_degrees(
        np.full(4, 174.0), np.full(4, -41.0),
        [174.5, 173.5, 174.0, 174.0], [-40.5, -41.5, -40.0, -42.0],
    )
    assert ((bearings >= 0) & (bearings < 360)).all()


# --- circular encoding -------------------------------------------------------

def test_circular_encoding_wraps_around():
    """The two ends of the range must land in the same place."""
    start = np.array(encode_circular(0, 24))
    full_turn = np.array(encode_circular(24, 24))
    assert np.allclose(start, full_turn)


def test_adjacent_hours_are_closer_than_opposite_hours():
    """23:00 and 01:00 are two hours apart, not twenty-two."""
    def separation(a, b):
        return np.hypot(*(np.array(encode_circular(a, 24)) - np.array(encode_circular(b, 24))))

    assert separation(23, 1) < separation(0, 12)


# --- local time --------------------------------------------------------------

def test_utc_converts_to_new_zealand_standard_time():
    """Midnight UTC in July is midday the same day in NZST, which is UTC+12."""
    frame = pd.DataFrame({"origin_time": ["2020-07-01T00:00:00Z"]})
    result = add_local_time_features(frame)
    assert result.loc[0, "local_hour"] == pytest.approx(12.0)


def test_daylight_saving_is_respected():
    """In January New Zealand is on NZDT, UTC+13, not UTC+12."""
    frame = pd.DataFrame({"origin_time": ["2020-01-01T00:00:00Z"]})
    result = add_local_time_features(frame)
    assert result.loc[0, "local_hour"] == pytest.approx(13.0)


def test_weekend_flag_uses_local_days():
    frame = pd.DataFrame({"origin_time": ["2020-07-04T00:00:00Z", "2020-07-06T00:00:00Z"]})
    result = add_local_time_features(frame)
    assert result.loc[0, "is_weekend"]        # Saturday
    assert not result.loc[1, "is_weekend"]    # Monday


def test_mixed_timestamp_formats_are_accepted():
    """GeoNet emits some times with fractional seconds and some without."""
    frame = pd.DataFrame({"origin_time": [
        "2016-11-13 11:02:56.346000+00:00",
        "2016-11-13 11:32:07+00:00",
    ]})
    result = add_local_time_features(frame)
    assert result["local_hour"].notna().all()


# --- vs30 --------------------------------------------------------------------

def make_cells():
    from clean import to_nztm
    easting, northing = to_nztm([CHRISTCHURCH[0]], [CHRISTCHURCH[1]])
    return pd.DataFrame({"cell_easting": easting, "cell_northing": northing})


def test_vs30_takes_the_nearest_sample():
    grid = pd.DataFrame({
        "longitude": [CHRISTCHURCH[0], WELLINGTON[0]],
        "latitude": [CHRISTCHURCH[1], WELLINGTON[1]],
        "vs30": [200.0, 700.0],
    })
    result = add_vs30(make_cells(), grid)
    assert result.loc[0, "vs30"] == 200.0


def test_vs30_is_missing_when_the_nearest_sample_is_too_far():
    """A value borrowed from 300 km away describes different ground."""
    grid = pd.DataFrame({
        "longitude": [WELLINGTON[0]], "latitude": [WELLINGTON[1]], "vs30": [700.0],
    })
    result = add_vs30(make_cells(), grid, max_match_km=10)
    assert np.isnan(result.loc[0, "vs30"])


def test_pipeline_survives_without_vs30():
    """The grid has unconfirmed licensing, so the project must run without it."""
    result = add_vs30(make_cells(), None)
    assert "vs30" in result.columns
    assert result["vs30"].isna().all()


def test_missing_vs30_file_returns_none():
    assert load_vs30_grid("does/not/exist.csv") is None


# --- assembled features ------------------------------------------------------

def test_hypocentral_distance_accounts_for_depth():
    """A cell directly above a 30 km deep event is 30 km from it, not zero."""
    cells = pd.DataFrame({
        "public_id": ["e"], "cell_x": [0], "cell_y": [0],
        "cell_easting": [1570717.0], "cell_northing": [5180163.0],
        "cell_longitude": [CHRISTCHURCH[0]], "cell_latitude": [CHRISTCHURCH[1]],
    })
    events = pd.DataFrame({
        "public_id": ["e"], "origin_time": ["2020-07-01T00:00:00Z"],
        "magnitude": [5.0], "depth_km": [30.0],
        "longitude": [CHRISTCHURCH[0]], "latitude": [CHRISTCHURCH[1]],
    })
    result = build_features(cells, events, vs30_grid=None, verbose=False)

    assert result.loc[0, "epicentral_distance_km"] == pytest.approx(0, abs=0.1)
    assert result.loc[0, "hypocentral_distance_km"] == pytest.approx(30, abs=0.1)
    assert result.loc[0, "log_hypocentral_distance"] == pytest.approx(np.log10(30), abs=1e-6)


# --- vs30 being optional, which the README promises --------------------------

def _cells(n=6):
    return pd.DataFrame({
        "cell_easting": np.linspace(1.6e6, 1.7e6, n),
        "cell_northing": np.linspace(5.4e6, 5.5e6, n),
    })


def test_no_vs30_grid_leaves_the_column_present_but_empty():
    result = add_vs30(_cells(), None)
    assert result["vs30"].isna().all()
    assert not result["vs30_imputed"].any()


def test_model_features_drops_a_column_with_no_data():
    """Without this the pipeline cannot run without vs30, which it advertises."""
    frame = add_vs30(_cells(), None).assign(
        magnitude=6.0, depth_km=15.0, log_hypocentral_distance=2.0,
        azimuth_sin=0.0, azimuth_cos=1.0, local_hour_sin=0.0,
        local_hour_cos=1.0, is_weekend=0)

    columns = model_features(frame, verbose=False)
    assert "vs30" not in columns
    assert set(columns) == set(MODEL_FEATURES) - {"vs30"}


def test_model_features_keeps_everything_when_the_data_is_there():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in MODEL_FEATURES})
    assert model_features(frame, verbose=False) == MODEL_FEATURES


def test_model_features_output_contains_no_missing_columns():
    """The contract: what comes back can be handed straight to an estimator."""
    frame = pd.DataFrame({column: [1.0, 2.0] for column in MODEL_FEATURES})
    frame["vs30"] = [np.nan, np.nan]

    for column in model_features(frame, verbose=False):
        assert frame[column].notna().all()


def test_a_partly_missing_vs30_is_filled_rather_than_left_to_break_a_fit():
    """A scattering of gaps is a coverage problem, not an absent feature."""
    cells = _cells(4)
    # One sample near the first two cells only, so the far ones exceed the cutoff.
    longitude, latitude = from_nztm(cells["cell_easting"][:1], cells["cell_northing"][:1])
    grid = pd.DataFrame({"longitude": longitude, "latitude": latitude, "vs30": [400.0]})

    result = add_vs30(cells, grid, max_match_km=5)

    assert result["vs30"].notna().all()      # nothing left for an estimator to trip on
    assert result["vs30_imputed"].any()      # but the fill is recorded
    assert not result["vs30_imputed"].all()
