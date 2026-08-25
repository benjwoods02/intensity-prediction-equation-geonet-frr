"""Tests for the gridding and target derivation in src/clean.py.

The statistics are checked against numpy operating on the expanded reports,
so the count-based shortcuts have to agree with the obvious slow method.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clean import (MMI_COLUMNS, MMI_LEVELS, aggregate_to_cells, assign_grid_cells,
                   classify_locations, filter_to_mainland, mmi_mean, mmi_median,
                   mmi_mode, mmi_mode_is_tied, to_nztm)


def make_felt(rows):
    """rows are (public_id, longitude, latitude, {mmi_level: count})."""
    records = []
    for public_id, longitude, latitude, counts in rows:
        record = {"public_id": public_id, "longitude": longitude, "latitude": latitude}
        for level in MMI_LEVELS:
            record[f"mmi_{level}"] = counts.get(level, 0)
        record["report_count"] = sum(counts.values())
        records.append(record)
    return pd.DataFrame(records)


def counts_array(*rows):
    return np.array(rows)


def expand(counts):
    """Turn a single row of level counts back into individual report values."""
    return np.repeat(MMI_LEVELS, counts)


# --- location classification -------------------------------------------------

def test_null_island_reports_are_identified():
    """Failed geolocation is published at (0.005, 0.003), carrying real counts."""
    felt = make_felt([("e", 0.005493, 0.002747, {4: 55})])
    assert classify_locations(felt).tolist() == ["null_island"]


def test_overseas_reports_are_identified():
    felt = make_felt([
        ("e", 151.2, -33.9, {3: 1}),    # Sydney
        ("e", -0.18, 51.5, {3: 1}),     # London
    ])
    assert classify_locations(felt).tolist() == ["elsewhere", "elsewhere"]


def test_mainland_reports_are_kept():
    felt = make_felt([("e", 172.64, -43.53, {4: 6})])
    assert classify_locations(felt).tolist() == ["mainland"]


def test_chathams_are_excluded_as_outside_the_projection():
    """Genuine NZ territory, but outside NZTM2000, so it cannot share the grid."""
    felt = make_felt([("e", -176.53, -43.96, {3: 2})])
    assert classify_locations(felt).tolist() == ["elsewhere"]


def test_filtering_removes_unusable_locations():
    felt = make_felt([
        ("e", 172.64, -43.53, {4: 6}),
        ("e", 0.005493, 0.002747, {4: 55}),
        ("e", 151.2, -33.9, {3: 1}),
    ])
    kept = filter_to_mainland(felt, verbose=False)
    assert len(kept) == 1
    assert kept.loc[0, "longitude"] == pytest.approx(172.64)


# --- projection and gridding -------------------------------------------------

def test_projection_matches_published_nztm_coordinates():
    """Christchurch Cathedral Square sits near E 1570000, N 5180000."""
    easting, northing = to_nztm(172.6376, -43.5309)
    assert easting == pytest.approx(1570717, abs=2000)
    assert northing == pytest.approx(5180163, abs=2000)


def test_grid_cells_are_one_kilometre_apart():
    """Two points 1 km apart in projected space must land in adjacent cells."""
    felt = make_felt([
        ("e", 172.6376, -43.5309, {4: 5}),
        ("e", 172.6376, -43.5309, {4: 5}),
    ])
    located = assign_grid_cells(felt, cell_size_m=1000)
    # Same point, so same cell
    assert located["cell_x"].nunique() == 1
    # Centroids sit at the half-cell offset
    assert located.loc[0, "cell_easting"] % 1000 == 500


def test_nearby_points_share_a_cell():
    """Points a few hundred metres apart should aggregate together."""
    felt = make_felt([
        ("e", 172.6376, -43.5309, {4: 5}),
        ("e", 172.6400, -43.5312, {4: 5}),
    ])
    located = assign_grid_cells(felt)
    assert len(located.groupby(["cell_x", "cell_y"])) == 1


# --- target statistics -------------------------------------------------------

@pytest.mark.parametrize("counts", [
    [3, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0],
    [1, 2, 3, 2, 1, 1],
    [0, 0, 0, 0, 0, 7],
    [0, 3, 0, 3, 0, 0],
])
def test_mean_matches_numpy_on_expanded_reports(counts):
    array = counts_array(counts)
    assert mmi_mean(array)[0] == pytest.approx(expand(counts).mean())


@pytest.mark.parametrize("counts", [
    [3, 0, 0, 0, 0, 0],
    [2, 1, 0, 0, 0, 0],
    [2, 2, 0, 0, 0, 0],
    [1, 2, 3, 2, 1, 1],
    [0, 0, 0, 0, 0, 7],
    [0, 3, 0, 3, 0, 0],
])
def test_median_matches_numpy_on_expanded_reports(counts):
    array = counts_array(counts)
    assert mmi_median(array)[0] == pytest.approx(np.median(expand(counts)))


def test_mode_returns_the_most_common_level():
    assert mmi_mode(counts_array([1, 2, 9, 0, 0, 0]))[0] == 5


def test_mode_ties_break_towards_lower_intensity():
    """A deliberate choice: overstating shaking is the worse error."""
    tied = counts_array([0, 3, 0, 3, 0, 0])  # MMI 4 and MMI 6 both have 3
    assert mmi_mode(tied)[0] == 4
    assert mmi_mode_is_tied(tied)[0]


def test_untied_mode_is_not_flagged():
    assert not mmi_mode_is_tied(counts_array([1, 5, 1, 0, 0, 0]))[0]


def test_median_can_land_between_levels():
    """An even report count averages the two central values."""
    assert mmi_median(counts_array([2, 2, 0, 0, 0, 0]))[0] == 3.5


# --- aggregation -------------------------------------------------------------

def test_reports_in_the_same_cell_are_summed():
    felt = make_felt([
        ("e", 172.6376, -43.5309, {3: 2}),
        ("e", 172.6400, -43.5312, {3: 1, 4: 3}),
    ])
    cells = aggregate_to_cells(felt, min_reports=1, verbose=False)
    assert len(cells) == 1
    assert cells.loc[0, "mmi_3"] == 3
    assert cells.loc[0, "mmi_4"] == 3
    assert cells.loc[0, "report_count"] == 6
    assert cells.loc[0, "locations"] == 2


def test_cells_below_the_minimum_are_dropped():
    felt = make_felt([
        ("e", 172.6376, -43.5309, {4: 2}),   # sparse, should go
        ("e", 174.7762, -41.2785, {4: 9}),   # well observed, should stay
    ])
    cells = aggregate_to_cells(felt, min_reports=5, verbose=False)
    assert len(cells) == 1
    assert cells.loc[0, "report_count"] == 9


def test_same_cell_different_earthquakes_stay_separate():
    """The grain is one row per earthquake per cell, not per cell."""
    felt = make_felt([
        ("quake_a", 172.6376, -43.5309, {4: 6}),
        ("quake_b", 172.6376, -43.5309, {6: 6}),
    ])
    cells = aggregate_to_cells(felt, min_reports=5, verbose=False)
    assert len(cells) == 2
    assert set(cells["public_id"]) == {"quake_a", "quake_b"}


def test_aggregated_counts_reconcile_with_report_count():
    felt = make_felt([
        ("e", 172.6376, -43.5309, {3: 2, 5: 4}),
        ("e", 174.7762, -41.2785, {6: 1, 8: 6}),
    ])
    cells = aggregate_to_cells(felt, min_reports=1, verbose=False)
    assert (cells[MMI_COLUMNS].sum(axis=1) == cells["report_count"]).all()
