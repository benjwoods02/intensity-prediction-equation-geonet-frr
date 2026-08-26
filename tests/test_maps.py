"""Tests for the shake map rendering in src/maps.py.

These maps are what caught the far field hole in the physical check, so they
are load-bearing rather than decorative and are worth testing like anything
else. The things that can silently go wrong are geometric: a land mask that
selects the sea, a grid reshaped along the wrong axis so the map is
transposed, or a colour scale that clips the failures it exists to show.

Rendering is exercised on a synthetic model into a temporary directory, so
nothing here touches the real dataset or the network.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import maps as MP
import spatial as S
from bench import Candidate
from features import MODEL_FEATURES


class FakeRegressor:
    """Intensity falling steadily with distance, as physics requires."""

    def predict(self, X):
        return 8.0 - 1.5 * np.asarray(X["log_hypocentral_distance"], dtype="float64")


def fake_candidate():
    """predict_field takes a bench Candidate, not a bare estimator."""
    return Candidate("Fake Model", FakeRegressor(), "regression", "test")


@pytest.fixture(scope="module")
def rings():
    return MP.load_coastline()


@pytest.fixture(scope="module")
def grid():
    # Coarse on purpose: these tests are about geometry, not resolution.
    return S.nz_grid(spacing_km=25)


# --- filenames ---------------------------------------------------------------

def test_slugify_survives_every_filesystem():
    assert MP.slugify("Hist Gradient Boosting (depth 4)") == "hist-gradient-boosting-depth-4"
    assert MP.slugify("MLP (128, 64)") == "mlp-128-64"


def test_slugify_has_no_leading_or_trailing_separator():
    assert not MP.slugify("(Bagging)").startswith("-")
    assert not MP.slugify("(Bagging)").endswith("-")


# --- the coastline -----------------------------------------------------------

def test_coastline_loads_as_closed_rings(rings):
    assert len(rings) >= 2  # at least the two main islands
    for ring in rings:
        assert ring.shape[1] == 2
        assert np.allclose(ring[0], ring[-1])  # closed


def test_coastline_covers_both_main_islands(rings):
    latitudes = np.concatenate([ring[:, 1] for ring in rings])
    assert latitudes.min() < -46      # Southland
    assert latitudes.max() > -35      # Northland


def test_coastline_excludes_the_chathams(rings):
    """They sit across the antimeridian and outside the NZTM grid."""
    longitudes = np.concatenate([ring[:, 0] for ring in rings])
    assert longitudes.min() > 0


# --- the land mask -----------------------------------------------------------

@pytest.mark.parametrize("place, longitude, latitude", [
    ("Wellington", 174.78, -41.29),
    ("Christchurch", 172.64, -43.53),
    ("Auckland", 174.76, -36.85),
    ("Dunedin", 170.50, -45.87),
    ("Gisborne", 178.02, -38.66),
    ("Invercargill", 168.35, -46.41),
])
def test_cities_are_on_land(place, longitude, latitude, rings):
    assert MP.land_mask(np.array([longitude]), np.array([latitude]), rings)[0], place


@pytest.mark.parametrize("place, longitude, latitude", [
    ("Tasman Sea", 170.0, -41.0),
    ("Pacific east of Gisborne", 179.0, -38.5),
    ("Southern Ocean", 172.0, -47.0),
])
def test_open_water_is_not_land(place, longitude, latitude, rings):
    assert not MP.land_mask(np.array([longitude]), np.array([latitude]), rings)[0], place


def test_mask_keeps_the_input_shape(rings):
    longitude = np.full((4, 5), 174.0)
    latitude = np.full((4, 5), -41.0)
    assert MP.land_mask(longitude, latitude, rings).shape == (4, 5)


def test_most_of_the_bounding_box_is_sea(grid, rings):
    """A mask that passed everything would silently disable itself."""
    on_land = MP.land_mask(grid["longitude"].to_numpy(), grid["latitude"].to_numpy(), rings)
    assert 0.05 < on_land.mean() < 0.45


# --- the grid ----------------------------------------------------------------

def test_grid_shape_matches_the_flattened_frame(grid):
    rows, columns = MP.grid_shape(grid)
    assert rows * columns == len(grid)


def test_field_is_not_transposed(grid, rings):
    """Reshaping along the wrong axis would rotate every map."""
    longitude, latitude, _ = MP.predict_field(fake_candidate(), MP.KAIKOURA, grid, rings,
                                              MODEL_FEATURES)
    # Longitude must vary across a row and latitude down a column.
    assert longitude[0].std() > 1.0
    assert longitude[:, 0].std() < 0.6
    assert latitude[:, 0].std() > 1.0


# --- predicted fields --------------------------------------------------------

def test_sea_is_blanked_and_land_is_not(grid, rings):
    _, _, intensity = MP.predict_field(fake_candidate(), MP.KAIKOURA, grid, rings,
                                       MODEL_FEATURES)
    assert np.isnan(intensity).any()
    assert np.isfinite(intensity).any()


def test_shaking_is_strongest_near_the_epicentre(grid, rings):
    longitude, latitude, intensity = MP.predict_field(fake_candidate(), MP.KAIKOURA,
                                                      grid, rings, MODEL_FEATURES)
    from features import haversine_km

    distance = haversine_km(MP.KAIKOURA["longitude"], MP.KAIKOURA["latitude"],
                            longitude, latitude)
    land = np.isfinite(intensity)
    near = intensity[land & (distance < 100)].mean()
    far = intensity[land & (distance > 500)].mean()
    assert near > far


def test_colour_range_spans_the_survey_scale():
    """Clipping at the top would hide exactly the failures the maps exist for."""
    assert MP.COLOUR_RANGE == (3.0, 8.0)


# --- rendering ---------------------------------------------------------------

@pytest.fixture
def one_row_verdict():
    import pandas as pd
    return pd.DataFrame([{"model": "Fake Model", "cell_mae": 0.35,
                          "worst_spearman": -0.98, "total_drop": 1.1,
                          "passes": True}])


def test_individual_maps_are_written(tmp_path, grid, rings, one_row_verdict):
    fields = {"Fake Model": MP.predict_field(fake_candidate(), MP.KAIKOURA, grid, rings,
                                             MODEL_FEATURES)}
    written = MP.individual_maps(fields, one_row_verdict, MP.KAIKOURA, rings, tmp_path)

    assert len(written) == 1
    assert written[0].exists() and written[0].stat().st_size > 0
    assert written[0].name == "01_fake-model.png"


def test_contact_sheet_is_written(tmp_path, grid, rings, one_row_verdict):
    fields = {"Fake Model": MP.predict_field(fake_candidate(), MP.KAIKOURA, grid, rings,
                                             MODEL_FEATURES)}
    path = MP.contact_sheet(fields, one_row_verdict, MP.KAIKOURA, rings,
                            tmp_path / "sheet.png")

    assert path.exists() and path.stat().st_size > 0


def test_comparison_needs_a_passing_model(tmp_path, grid, rings, one_row_verdict):
    """It contrasts the selection against the rejection, so both must exist."""
    fields = {"Fake Model": MP.predict_field(fake_candidate(), MP.KAIKOURA, grid, rings,
                                             MODEL_FEATURES)}
    failing = one_row_verdict.assign(passes=False)

    with pytest.raises(IndexError):
        MP.comparison(fields, failing, MP.KAIKOURA, rings, tmp_path / "x.png")


# --- paths -------------------------------------------------------------------

def test_paths_are_anchored_to_the_repository_not_the_working_directory():
    """Running the module from elsewhere must not change where it reads or writes."""
    assert MP.REPO_ROOT.is_absolute()
    assert MP.COASTLINE_PATH.is_absolute()
    assert MP.COASTLINE_PATH.exists()
