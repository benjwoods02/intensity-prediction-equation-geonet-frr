"""Aggregate felt report locations onto a 1 km grid and derive the MMI target.

Two things happen here.

Spatial aggregation. GeoNet publishes felt reports as points, but individual
points are noisy: one person's answer at one address carries no weight on its
own. Grouping reports into 1 km cells and requiring a minimum number per cell
trades spatial resolution for a target that means something.

The grid is built in NZTM2000 (EPSG:2193) rather than in degrees. A degree of
latitude is roughly 111 km everywhere, but a degree of longitude shrinks
towards the poles, so a grid defined in degrees would produce cells that are
noticeably narrower in Southland than in Northland. Projecting to metres first
gives cells that are genuinely 1 km on a side.

Target definition. Each cell holds a distribution of MMI values rather than a
single number, so a central tendency measure has to be chosen. Mean, median
and mode are all computed and carried forward together, so the choice can be
made with evidence in the exploratory notebook rather than being fixed here.

All three are derived directly from the MMI level counts rather than by
expanding the counts back into individual rows. The results are identical and
it avoids materialising roughly 600,000 rows.
"""

import numpy as np
import pandas as pd
from pyproj import Transformer

WGS84 = "EPSG:4326"
NZTM = "EPSG:2193"

MMI_LEVELS = np.arange(3, 9)
MMI_COLUMNS = [f"mmi_{level}" for level in MMI_LEVELS]

# Mainland New Zealand, which is also roughly the extent over which NZTM2000
# is valid. The Chatham Islands sit outside it, on the far side of the
# antimeridian, and use their own projection.
MAINLAND_BOUNDS = {
    "min_longitude": 166.0,
    "max_longitude": 179.0,
    "min_latitude": -47.5,
    "max_latitude": -34.0,
}

# Reports whose geolocation failed are published at this exact coordinate,
# a few hundred metres from the origin of the coordinate system.
NULL_ISLAND_TOLERANCE = 0.5

_to_nztm = Transformer.from_crs(WGS84, NZTM, always_xy=True)
_from_nztm = Transformer.from_crs(NZTM, WGS84, always_xy=True)


def _as_coordinate_input(values):
    """Coerce coordinates into a form pyproj transforms without complaint.

    pyproj routes a single-element numpy array through its scalar point
    transform, which numpy deprecates and will eventually reject. Multi-element
    arrays and plain lists both take the vectorised path, so a one-element
    array is handed over as a list. At that size the conversion costs nothing.
    """
    array = np.atleast_1d(np.asarray(values, dtype="float64"))
    return array.tolist() if array.size == 1 else array


def _transform(transformer, first, second):
    """Run a pyproj transform and always return numpy arrays."""
    out_first, out_second = transformer.transform(
        _as_coordinate_input(first), _as_coordinate_input(second)
    )
    return np.atleast_1d(np.asarray(out_first)), np.atleast_1d(np.asarray(out_second))


def classify_locations(felt):
    """Label each reporting location as mainland, null island, or elsewhere.

    Kept separate from filtering so the notebook can report exactly what was
    removed and why, rather than a single opaque row count.
    """
    longitude = felt["longitude"]
    latitude = felt["latitude"]

    null_island = (longitude.abs() < NULL_ISLAND_TOLERANCE) & (latitude.abs() < NULL_ISLAND_TOLERANCE)

    mainland = (
        longitude.between(MAINLAND_BOUNDS["min_longitude"], MAINLAND_BOUNDS["max_longitude"])
        & latitude.between(MAINLAND_BOUNDS["min_latitude"], MAINLAND_BOUNDS["max_latitude"])
    )

    category = pd.Series("elsewhere", index=felt.index, dtype="object")
    category[mainland] = "mainland"
    category[null_island] = "null_island"

    return category


def filter_to_mainland(felt, verbose=True):
    """Drop reporting locations that cannot be used, and say what went.

    Three groups are removed:

      null_island  Reports whose geolocation failed, published at (0.005, 0.003).
                   These carry real report counts, up to 55 at a single point,
                   so leaving them in would invent a densely observed cell
                   thousands of kilometres from every epicentre.

      elsewhere    Genuine reports filed from outside New Zealand: Australia,
                   the United Kingdom, the Philippines and others. Real people,
                   but not New Zealand shaking, and the distances would be
                   meaningless.

      chathams     A small number of legitimate Chatham Islands reports. These
                   are dropped only because they fall outside the NZTM2000 zone
                   of validity, so they cannot share a projected grid with the
                   mainland. Worth stating as a known limitation.
    """
    category = classify_locations(felt)
    kept = felt[category == "mainland"].reset_index(drop=True)

    if verbose:
        total_reports = int(felt["report_count"].sum())
        for label in ["null_island", "elsewhere"]:
            subset = felt[category == label]
            if len(subset):
                print(
                    f"  dropped {len(subset):,} {label} locations "
                    f"carrying {int(subset['report_count'].sum()):,} reports"
                )
        dropped_reports = total_reports - int(kept["report_count"].sum())
        print(
            f"  kept {len(kept):,} of {len(felt):,} locations "
            f"({dropped_reports:,} of {total_reports:,} reports removed, "
            f"{100 * dropped_reports / total_reports:.2f}%)"
        )

    return kept


def to_nztm(longitude, latitude):
    """Project WGS84 coordinates to NZTM2000 eastings and northings, in metres."""
    return _transform(_to_nztm, longitude, latitude)


def from_nztm(easting, northing):
    """Project NZTM2000 coordinates back to WGS84 longitude and latitude."""
    return _transform(_from_nztm, easting, northing)


def assign_grid_cells(felt, cell_size_m=1000):
    """Attach a projected grid cell index and centroid to each reporting location.

    Cell indices are the floor of the projected coordinate divided by the cell
    size, so every point inside a cell maps to the same pair of integers. The
    centroid is used later for distance calculations, so that every report in a
    cell is treated as being the same distance from the epicentre.
    """
    easting, northing = to_nztm(felt["longitude"], felt["latitude"])

    cell_x = np.floor(easting / cell_size_m).astype("int64")
    cell_y = np.floor(northing / cell_size_m).astype("int64")

    return felt.assign(
        easting=easting,
        northing=northing,
        cell_x=cell_x,
        cell_y=cell_y,
        cell_easting=(cell_x + 0.5) * cell_size_m,
        cell_northing=(cell_y + 0.5) * cell_size_m,
    )


def mmi_mean(counts):
    """Report-weighted mean MMI for each row of an MMI count matrix."""
    totals = counts.sum(axis=1)
    return (counts * MMI_LEVELS).sum(axis=1) / totals


def mmi_mode(counts):
    """Most frequently reported MMI for each row.

    Ties are broken towards the lower intensity. argmax returns the first
    maximum and the levels are in ascending order, so this happens naturally,
    but it is a deliberate choice: overstating shaking is the worse error.
    """
    return MMI_LEVELS[np.argmax(counts, axis=1)]


def mmi_mode_is_tied(counts):
    """Flag rows where two or more MMI levels share the top count."""
    maxima = counts.max(axis=1, keepdims=True)
    return (counts == maxima).sum(axis=1) > 1


def mmi_median(counts):
    """Median MMI for each row, using the conventional definition.

    With an even number of reports the two central values are averaged, so
    this can return a half step such as 4.5. That is intentional: it keeps the
    median comparable with the mean in the central tendency comparison, rather
    than silently rounding towards one side.
    """
    totals = counts.sum(axis=1)
    cumulative = counts.cumsum(axis=1)

    lower_position = (totals + 1) // 2
    upper_position = (totals + 2) // 2

    lower_index = (cumulative < lower_position[:, None]).sum(axis=1)
    upper_index = (cumulative < upper_position[:, None]).sum(axis=1)

    return (MMI_LEVELS[lower_index] + MMI_LEVELS[upper_index]) / 2


def aggregate_to_cells(felt, min_reports=5, cell_size_m=1000, verbose=True):
    """Aggregate reporting locations to one row per earthquake per grid cell.

    Report counts at each MMI level are summed across every location falling in
    the cell, then the three candidate targets are derived from that combined
    distribution. Cells below min_reports are dropped: a cell holding two
    reports gives a target that is mostly noise.
    """
    located = assign_grid_cells(felt, cell_size_m=cell_size_m)

    grouped = (
        located
        .groupby(["public_id", "cell_x", "cell_y"], as_index=False)
        .agg(
            cell_easting=("cell_easting", "first"),
            cell_northing=("cell_northing", "first"),
            locations=("report_count", "size"),
            **{column: (column, "sum") for column in MMI_COLUMNS},
        )
    )

    grouped["report_count"] = grouped[MMI_COLUMNS].sum(axis=1)

    before = len(grouped)
    grouped = grouped[grouped["report_count"] >= min_reports].reset_index(drop=True)

    counts = grouped[MMI_COLUMNS].to_numpy()

    grouped["mmi_mean"] = mmi_mean(counts)
    grouped["mmi_median"] = mmi_median(counts)
    grouped["mmi_mode"] = mmi_mode(counts)
    grouped["mode_is_tied"] = mmi_mode_is_tied(counts)

    longitude, latitude = from_nztm(grouped["cell_easting"], grouped["cell_northing"])
    grouped["cell_longitude"] = longitude
    grouped["cell_latitude"] = latitude

    if verbose:
        print(f"  {before:,} cells before the minimum report rule")
        print(f"  {len(grouped):,} cells with at least {min_reports} reports "
              f"({100 * len(grouped) / before:.1f}% kept)")
        print(f"  {int(grouped['report_count'].sum()):,} reports retained")
        print(f"  {grouped['mode_is_tied'].sum():,} cells have a tied mode "
              f"({100 * grouped['mode_is_tied'].mean():.1f}%)")

    return grouped


def build_cells(felt, min_reports=5, cell_size_m=1000, verbose=True):
    """Run the full Phase 2 pipeline: filter, project, grid, aggregate."""
    if verbose:
        print("Filtering reporting locations")
    filtered = filter_to_mainland(felt, verbose=verbose)

    if verbose:
        print(f"Aggregating to a {cell_size_m} m grid")
    return aggregate_to_cells(
        filtered, min_reports=min_reports, cell_size_m=cell_size_m, verbose=verbose
    )


def expand_to_weighted_labels(cells, feature_columns=None):
    """Turn each cell into one weighted row per MMI level actually reported.

    Rather than collapsing a cell to a single summary value, every reported
    level is kept and carries the number of people who chose it as its weight.
    A cell where 12 people said MMI 4 and 9 said MMI 5 contributes both rows,
    weighted 12 and 9, instead of being flattened to "4".

    This matters because collapsing loses a lot. The mode accounts for a median
    of only about 55% of a cell's reports, and in roughly a third of cells it is
    a minority view that most respondents disagreed with. Weighting also removes
    the tie-breaking problem entirely: there is no longer a single winner to
    pick, so the 12.5% of cells with a tied mode stop being arbitrary.

    Every label is a genuine integer report, so the target never takes an
    impossible half-step value. The model still yields a continuous surface,
    because the probability distribution it predicts can be collapsed to an
    expected MMI when one is needed for mapping.

    Returns one row per (cell, reported level), with columns:
      mmi     the reported intensity, an integer between 3 and 8
      weight  how many people at that cell reported it
    """
    identifiers = ["public_id", "cell_x", "cell_y"]
    carried = list(feature_columns) if feature_columns is not None else [
        column for column in cells.columns
        if column not in MMI_COLUMNS and column not in identifiers
    ]

    long_form = cells.melt(
        id_vars=identifiers + carried,
        value_vars=MMI_COLUMNS,
        var_name="mmi",
        value_name="weight",
    )

    long_form["mmi"] = long_form["mmi"].str.removeprefix("mmi_").astype("int64")

    # A level nobody reported contributes nothing and would only add rows a
    # weighted fit has to ignore.
    return long_form[long_form["weight"] > 0].reset_index(drop=True)
