"""Derive modelling features for each earthquake and grid cell pair.

Three groups of feature are built here.

Physical. Distance from the source, magnitude and depth are what any
attenuation relation is built on. Distance is measured to the hypocentre
rather than the epicentre, because a shallow event and a deep one at the same
map distance do not shake the surface equally, and it is taken as a base-ten
logarithm because intensity falls off roughly with the logarithm of distance.

Directional. Rupture is not symmetric, so shaking can be stronger along the
direction a fault tears than across it. Azimuth from the epicentre to the cell
captures that possibility.

Reporting behaviour. Felt reports are filed by people, so the data records not
only how hard the ground shook but how many people were awake, indoors and
inclined to fill in a survey. Time of day and whether it was a weekend are
included to give the model a chance to account for that, rather than letting
it leak into the physical terms.

Both azimuth and time of day are circular: 359 degrees is adjacent to 1
degree, and 23:00 is adjacent to 00:00. Feeding either in as a raw number
would tell a model that those pairs are maximally far apart. Each is therefore
encoded as a sine and cosine pair, which preserves the wrap-around.
"""

from datetime import timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from clean import to_nztm

EARTH_RADIUS_KM = 6371.0088
NZ_TIMEZONE = ZoneInfo("Pacific/Auckland")

# Beyond this, the nearest known vs30 sample is too far away to describe the
# ground under a cell. The source grid is roughly 5 km spaced.
VS30_MAX_MATCH_KM = 10.0


def haversine_km(longitude_a, latitude_a, longitude_b, latitude_b):
    """Great-circle distance in kilometres between two sets of coordinates.

    Used in preference to straight-line distance in projected space so the
    result does not depend on the projection being valid at either point.
    """
    lon_a, lat_a, lon_b, lat_b = map(
        lambda values: np.radians(np.asarray(values, dtype="float64")),
        (longitude_a, latitude_a, longitude_b, latitude_b),
    )

    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a

    inner = np.sin(delta_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))


def initial_bearing_degrees(longitude_a, latitude_a, longitude_b, latitude_b):
    """Compass bearing from point A to point B, in degrees clockwise from north.

    This is the forward azimuth along a great circle. Over New Zealand the
    difference from a simple projected bearing is negligible, but it costs
    nothing to be correct.
    """
    lon_a, lat_a, lon_b, lat_b = map(
        lambda values: np.radians(np.asarray(values, dtype="float64")),
        (longitude_a, latitude_a, longitude_b, latitude_b),
    )

    delta_lon = lon_b - lon_a
    x = np.sin(delta_lon) * np.cos(lat_b)
    y = np.cos(lat_a) * np.sin(lat_b) - np.sin(lat_a) * np.cos(lat_b) * np.cos(delta_lon)

    return np.degrees(np.arctan2(x, y)) % 360


def encode_circular(values, period):
    """Encode a cyclic quantity as a sine and cosine pair.

    A raw angle or hour tells a model that the two ends of the range are far
    apart when they are in fact adjacent. Projecting onto a circle removes the
    artificial discontinuity.
    """
    radians = 2 * np.pi * np.asarray(values, dtype="float64") / period
    return np.sin(radians), np.cos(radians)


def add_local_time_features(frame, time_column="origin_time"):
    """Add local time of day and weekend features from a UTC timestamp.

    Converted to New Zealand local time rather than left in UTC, because what
    matters is whether people were awake where the shaking happened. The
    conversion respects daylight saving.
    """
    # GeoNet timestamps are inconsistent: some carry fractional seconds and
    # some do not, so a single format string fails once the frame has been
    # round-tripped through CSV and the values are strings again.
    timestamps = pd.to_datetime(frame[time_column], utc=True, format="mixed")
    local = timestamps.dt.tz_convert(NZ_TIMEZONE)

    hour = local.dt.hour + local.dt.minute / 60
    hour_sin, hour_cos = encode_circular(hour, period=24)

    return frame.assign(
        local_hour=hour,
        local_hour_sin=hour_sin,
        local_hour_cos=hour_cos,
        is_weekend=local.dt.dayofweek.isin([5, 6]),
    )


def load_vs30_grid(path):
    """Read a vs30 point grid, accepting either WKT geometry or plain columns.

    Returns None if the file is absent, so the pipeline can run without it.
    Vs30 describes how soft the ground is, which affects how much shaking is
    amplified, but the grid used here comes from a third-party source whose
    redistribution terms are unconfirmed. Treating it as optional keeps the
    rest of the project reproducible by anyone.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None

    grid = pd.read_csv(path)

    if "geometry" in grid.columns and "longitude" not in grid.columns:
        coordinates = grid["geometry"].str.extract(r"POINT \(([-\d.eE+]+) ([-\d.eE+]+)\)")
        grid["longitude"] = pd.to_numeric(coordinates[0], errors="coerce")
        grid["latitude"] = pd.to_numeric(coordinates[1], errors="coerce")

    grid = grid.dropna(subset=["longitude", "latitude", "vs30"])
    return grid[["longitude", "latitude", "vs30"]].reset_index(drop=True)


def add_vs30(cells, vs30_grid, max_match_km=VS30_MAX_MATCH_KM):
    """Attach the vs30 value of the nearest sample point to each cell.

    Matching happens in projected space so that the search distance is in real
    metres. Cells whose nearest sample is further away than max_match_km get a
    missing value rather than a misleading one borrowed from far off.
    """
    if vs30_grid is None or vs30_grid.empty:
        return cells.assign(vs30=np.nan)

    from scipy.spatial import cKDTree

    grid_easting, grid_northing = to_nztm(vs30_grid["longitude"], vs30_grid["latitude"])
    tree = cKDTree(np.column_stack([grid_easting, grid_northing]))

    distances_m, indices = tree.query(
        np.column_stack([cells["cell_easting"], cells["cell_northing"]])
    )

    values = vs30_grid["vs30"].to_numpy()[indices]
    values = np.where(distances_m / 1000 <= max_match_km, values, np.nan)

    return cells.assign(vs30=values, vs30_match_km=distances_m / 1000)


def build_features(cells, events, vs30_grid=None, verbose=True):
    """Join event attributes onto cells and derive every modelling feature."""
    event_columns = ["public_id", "origin_time", "magnitude", "depth_km", "longitude", "latitude"]
    merged = cells.merge(
        events[event_columns].rename(
            columns={"longitude": "epicentre_longitude", "latitude": "epicentre_latitude"}
        ),
        on="public_id",
        how="inner",
    )

    if verbose and len(merged) != len(cells):
        print(f"  {len(cells) - len(merged):,} cells dropped: no matching event record")

    merged["epicentral_distance_km"] = haversine_km(
        merged["epicentre_longitude"], merged["epicentre_latitude"],
        merged["cell_longitude"], merged["cell_latitude"],
    )

    # Pythagoras through the earth: a cell directly above a 15 km deep event is
    # 15 km from it, not zero.
    merged["hypocentral_distance_km"] = np.sqrt(
        merged["epicentral_distance_km"] ** 2 + merged["depth_km"] ** 2
    )
    merged["log_hypocentral_distance"] = np.log10(merged["hypocentral_distance_km"])

    azimuth = initial_bearing_degrees(
        merged["epicentre_longitude"], merged["epicentre_latitude"],
        merged["cell_longitude"], merged["cell_latitude"],
    )
    azimuth_sin, azimuth_cos = encode_circular(azimuth, period=360)
    merged["azimuth_degrees"] = azimuth
    merged["azimuth_sin"] = azimuth_sin
    merged["azimuth_cos"] = azimuth_cos

    merged = add_local_time_features(merged)
    merged = add_vs30(merged, vs30_grid)

    if verbose:
        covered = merged["vs30"].notna().mean() * 100
        print(f"  {len(merged):,} rows with features")
        print(f"  hypocentral distance {merged['hypocentral_distance_km'].min():.1f} to "
              f"{merged['hypocentral_distance_km'].max():.0f} km")
        print(f"  vs30 coverage: {covered:.1f}% of cells")

    return merged


MODEL_FEATURES = [
    "magnitude",
    "depth_km",
    "log_hypocentral_distance",
    "azimuth_sin",
    "azimuth_cos",
    "local_hour_sin",
    "local_hour_cos",
    "is_weekend",
    "vs30",
]
