"""Aggregate felt report locations onto a 1 km grid and derive the MMI target.

Two things happen here.

Spatial aggregation. GeoNet publishes felt reports as points, but individual
points are noisy: one person's answer at one address carries no weight on its
own. Grouping reports into 1 km cells and requiring a minimum number per cell
reduces noise.

The grid is built in NZTM2000 (EPSG:2193).

Target definition. Each cell holds a distribution of MMI values rather than a
single number, so a central tendency measure has to be chosen. Mean, median
and mode are all computed and carried forward together, so the choice can be
made with evidence in the exploratory notebook rather than being fixed here.
"""

import numpy as np
import pandas as pd
from pyproj import Transformer

WGS84 = "EPSG:4326"
NZTM = "EPSG:2193"

MMI_LEVELS = np.arange(3, 9)
MMI_COLUMNS = [f"mmi_{level}" for level in MMI_LEVELS]

MAINLAND_BOUNDS = {
    "min_longitude": 166.0,
    "max_longitude": 179.0,
    "min_latitude": -47.5,
    "max_latitude": -34.0,
}

NULL_ISLAND_TOLERANCE = 0.5

_to_nztm = Transformer.from_crs(WGS84, NZTM, always_xy=True)
_from_nztm = Transformer.from_crs(NZTM, WGS84, always_xy=True)


def _as_coordinate_input(values):
    """Coerce coordinates into a form pyproj transforms.
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
    """Drop reporting locations that cannot be used.

    Three groups are removed:

      null_island  Reports whose geolocation failed, published at (0.005, 0.003).

      elsewhere    Everything else outside the mainland box.
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
    Ties are broken towards the lower intensity.
    """
    return MMI_LEVELS[np.argmax(counts, axis=1)]


def mmi_mode_is_tied(counts):
    """Flag rows where two or more MMI levels share the top count."""
    maxima = counts.max(axis=1, keepdims=True)
    return (counts == maxima).sum(axis=1) > 1


def mmi_median(counts):
    """Median MMI for each row, using the conventional definition.
    With an even number of reports the two central values are averaged, so
    this can return a half step such as 4.5.
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


WEIGHT_SCHEMES = ("sqrt", "count", "equal")


def apply_weight_scheme(long_form, scheme="sqrt"):
    """Set how much each cell contributes overall, without changing its label mix.

    Within a cell the relative weights always follow the reported proportions,
    so the label distribution is untouched. What changes is how much the cell
    counts against every other cell:

      count   influence proportional to the number of reports. This is would
              be good choice if reports were independent observations.

      sqrt    influence proportional to the square root of the report count.

      equal   every cell contributes exactly one unit regardless of how many
              people reported. Removes population bias entirely, at the cost
              of treating a five report cell as being as reliable as a five
              hundred report one.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight scheme {scheme!r}, expected one of {WEIGHT_SCHEMES}")

    totals = long_form.groupby(["public_id", "cell_x", "cell_y"])["weight"].transform("sum")
    proportion = long_form["weight"] / totals

    if scheme == "count":
        influence = totals
    elif scheme == "sqrt":
        influence = np.sqrt(totals)
    else:
        influence = 1.0

    return long_form.assign(weight=proportion * influence, cell_reports=totals)


def expand_to_weighted_labels(cells, feature_columns=None):
    """Turn each cell into one weighted row per MMI level actually reported.

    Returns one row per (cell, reported level), with columns:
      mmi     the reported intensity, an integer between 3 and 8
      weight  how many people at that cell reported it
    """
    identifiers = ["public_id", "cell_x", "cell_y"]
    requested = list(feature_columns) if feature_columns is not None else [
        column for column in cells.columns
        if column not in MMI_COLUMNS and column not in identifiers
    ]


    seen = set(identifiers)
    carried = []
    for column in requested:
        if column not in seen:
            carried.append(column)
            seen.add(column)

    long_form = cells.melt(
        id_vars=identifiers + carried,
        value_vars=MMI_COLUMNS,
        var_name="mmi",
        value_name="weight",
    )

    long_form["mmi"] = long_form["mmi"].str.removeprefix("mmi_").astype("int64")

    return long_form[long_form["weight"] > 0].reset_index(drop=True)
