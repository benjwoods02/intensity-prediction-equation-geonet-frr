"""Derive modelling features for each earthquake and grid cell pair.

Three groups of feature are built here.

Physical:
    -Distance from the source. Measured to the hypocentre. Taken as a base-ten
    log
    -Magnitude
    -Depth

Directional:
    -Azimuth from epicenter to cell captures asymmetric rupture.

Reporting behaviour: Felt reports are filed by people, so the data records not
only how hard the ground shook but how many people were awake, and
inclined to fill in a survey.
    -Time of day
    -Weekend or not

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

    This is the forward azimuth along a great circle.
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
    """
    radians = 2 * np.pi * np.asarray(values, dtype="float64") / period
    return np.sin(radians), np.cos(radians)


def add_local_time_features(frame, time_column="origin_time"):
    """Add local time of day and weekend features from a UTC timestamp.
    This is for reporting behaviour features.
    """
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
    metres. Cells whose nearest sample is further away than max_match_km get an
    average value.
    """
    if vs30_grid is None or vs30_grid.empty:
        return cells.assign(vs30=np.nan, vs30_match_km=np.nan, vs30_imputed=False)

    from scipy.spatial import cKDTree

    grid_easting, grid_northing = to_nztm(vs30_grid["longitude"], vs30_grid["latitude"])
    tree = cKDTree(np.column_stack([grid_easting, grid_northing]))

    distances_m, indices = tree.query(
        np.column_stack([cells["cell_easting"], cells["cell_northing"]])
    )

    values = vs30_grid["vs30"].to_numpy()[indices]
    values = np.where(distances_m / 1000 <= max_match_km, values, np.nan)

    matched = np.isfinite(values)
    imputed = ~matched
    if matched.any() and imputed.any():
        values = np.where(matched, values, float(np.median(values[matched])))

    return cells.assign(vs30=values, vs30_match_km=distances_m / 1000,
                        vs30_imputed=imputed)


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
        filled = int(merged["vs30_imputed"].sum())
        if filled:
            print(f"  {filled:,} cells had no vs30 sample within "
                  f"{VS30_MAX_MATCH_KM:.0f} km and were filled with the median")
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


def model_features(frame, columns=MODEL_FEATURES, verbose=True):
    """The subset of MODEL_FEATURES a given frame can actually supply.

    Vs30 is optional here.
    """
    usable = [column for column in columns
              if column in frame.columns and frame[column].notna().any()]
    dropped = [column for column in columns if column not in usable]

    if dropped and verbose:
        print(f"  dropped, no data available: {', '.join(dropped)}")

    return usable