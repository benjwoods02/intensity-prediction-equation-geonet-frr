"""Fetch earthquake events and Felt RAPID Report data from the public GeoNet APIs.

Two endpoints are used:

  quakesearch.geonet.org.nz/geojson
      Event catalogue. Returns one feature per earthquake with magnitude,
      depth, origin time and location.

  api.geonet.org.nz/intensity?type=reported
      Felt reports for a single event. GeoNet aggregates individual public
      submissions to points before publishing them, so each feature is a
      location carrying a report count and, usefully, count_mmi: the full
      distribution of MMI values reported at that point. Keeping that
      distribution rather than a pre-computed summary is what lets the
      central tendency measure (mean, median or mode) be chosen later
      rather than being baked in here.

Responses are cached to data/raw so repeated runs do not hit the API again.
Delete the cache directory to force a refresh.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

QUAKE_SEARCH_URL = "https://quakesearch.geonet.org.nz/geojson"
INTENSITY_URL = "https://api.geonet.org.nz/intensity"

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EVENTS_DIR = RAW_DIR / "events"
FELT_DIR = RAW_DIR / "felt"

# The Felt RAPID Report survey presents six cartoons corresponding to MMI 3
# through 8, so that is the full range of values the data should contain.
MMI_LEVELS = range(3, 9)

REQUEST_TIMEOUT = 60
REQUEST_DELAY = 0.5  # be polite to a free public API


def _get_json(url, params, cache_path, force=False):
    """GET a JSON response, using the cached copy unless force is set."""
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(REQUEST_DELAY)

    return payload


def fetch_events(min_magnitude, max_magnitude, start_date, end_date, force=False):
    """Return a DataFrame of earthquake events in the given magnitude and date range.

    Dates are ISO strings, e.g. "2016-01-01T00:00:00". The GeoNet catalogue
    covers all of New Zealand; filtering to events that actually generated
    felt reports happens later, once report counts are known.
    """
    params = {
        "minmag": min_magnitude,
        "maxmag": max_magnitude,
        "startdate": start_date,
        "enddate": end_date,
    }
    # Formatted rather than interpolated raw, so that fetch_events(6, 8, ...)
    # and fetch_events(6.0, 8.0, ...) resolve to the same cache file instead of
    # fetching the identical query twice under two names.
    cache_name = (f"events_{float(min_magnitude):.1f}_{float(max_magnitude):.1f}"
                  f"_{start_date[:10]}_{end_date[:10]}.json")
    payload = _get_json(QUAKE_SEARCH_URL, params, EVENTS_DIR / cache_name, force=force)

    events = []
    for feature in payload.get("features", []):
        properties = feature["properties"]
        longitude, latitude = feature["geometry"]["coordinates"][:2]
        events.append({
            "public_id": properties["publicid"],
            "origin_time": properties["origintime"],
            "magnitude": properties["magnitude"],
            "depth_km": properties["depth"],
            "longitude": longitude,
            "latitude": latitude,
        })

    frame = pd.DataFrame(events)
    if not frame.empty:
        frame["origin_time"] = pd.to_datetime(frame["origin_time"], format="mixed", utc=True)
        frame = frame.sort_values("origin_time").reset_index(drop=True)

    return frame


def filter_to_new_zealand(events):
    """Drop events outside New Zealand.

    The GeoNet catalogue includes teleseismic events: distant earthquakes
    picked up by New Zealand instruments. A November 2016 query returns
    events in Chile and Japan alongside the Kaikoura sequence. They are real
    earthquakes but not New Zealand ones, and leaving them in would produce
    meaningless hypocentral distances.

    The bounding box is split because New Zealand straddles the antimeridian:
    the mainland sits around 165 to 180 degrees east, while the Chatham
    Islands sit just past it at roughly -177 degrees.
    """
    if events.empty:
        return events

    in_latitude = events["latitude"].between(-48.5, -33.0)
    mainland = events["longitude"].between(165.0, 180.0)
    chathams = events["longitude"].between(-180.0, -175.0)

    return events[in_latitude & (mainland | chathams)].reset_index(drop=True)


def drop_duplicate_events(events):
    """Collapse repeated catalogue entries for the same physical earthquake.

    GeoNet sometimes publishes several entries sharing an origin time and
    location, differing only in magnitude solution. The highest magnitude is
    kept, on the basis that later revisions tend to refine upward for large
    events.

    On the current New Zealand window this removes nothing, and it is kept as a
    guard rather than claimed as a finding. The duplicates that do exist in the
    raw catalogue, such as the three November 2016 entries off Fukushima, are
    all teleseismic, so filter_to_new_zealand reaches them first. A different
    date range or a change at GeoNet could put duplicates inside the New
    Zealand box, and silently training on the same earthquake twice would be
    worse than an inert filter.
    """
    if events.empty:
        return events

    return (
        events
        .sort_values("magnitude", ascending=False)
        .drop_duplicates(subset=["origin_time", "longitude", "latitude"], keep="first")
        .sort_values("origin_time")
        .reset_index(drop=True)
    )


def assign_magnitude_bins(events, bin_width=0.5):
    """Label each event with the magnitude bin it falls into.

    Bins are left-closed, so a magnitude of exactly 5.0 with a width of 0.5
    lands in "5.0-5.5" rather than "4.5-5.0".
    """
    if events.empty:
        return events.assign(magnitude_bin=pd.Series(dtype="object"))

    lower = (events["magnitude"] // bin_width) * bin_width
    labels = lower.map(lambda x: f"{x:.1f}-{x + bin_width:.1f}")

    return events.assign(magnitude_bin=labels)


def select_stratified_events(events, bin_width=0.5, max_per_bin=None, random_state=7):
    """Sample events evenly across magnitude bins.

    Earthquake catalogues are dominated by small events, so an unstratified
    sample would be almost entirely low magnitude and the model would barely
    see the large shaking it most needs to predict. Sampling within bins
    trades total volume for coverage across the magnitude range.

    Bins holding fewer than max_per_bin events contribute everything they
    have, so the result is balanced only as far as the catalogue allows.
    Passing max_per_bin=None keeps every event and just adds the bin labels.
    """
    binned = assign_magnitude_bins(events, bin_width=bin_width)

    if max_per_bin is None or binned.empty:
        return binned.reset_index(drop=True)

    # An explicit loop rather than groupby.apply: newer pandas drops the
    # grouping column from the frames passed to apply, which silently loses
    # magnitude_bin from the result.
    parts = [
        group.sample(min(len(group), max_per_bin), random_state=random_state)
        for _, group in binned.groupby("magnitude_bin", observed=True)
    ]

    return pd.concat(parts).sort_values("origin_time").reset_index(drop=True)


def summarise_bins(events):
    """Count events per magnitude bin, for checking how balanced a selection is."""
    if events.empty or "magnitude_bin" not in events.columns:
        return pd.DataFrame(columns=["magnitude_bin", "events"])

    return (
        events.groupby("magnitude_bin")
        .size()
        .reset_index(name="events")
        .sort_values("magnitude_bin")
        .reset_index(drop=True)
    )


def fetch_felt_reports(public_id, force=False, warn_out_of_range=True):
    """Return a DataFrame of felt report points for one earthquake.

    One row per reporting location, with the MMI distribution at that
    location expanded into columns named mmi_3 through mmi_8. Events with no
    felt reports return an empty DataFrame rather than raising.
    """
    payload = _get_json(
        INTENSITY_URL,
        {"type": "reported", "publicID": public_id},
        FELT_DIR / f"{public_id}.json",
        force=force,
    )

    rows = []
    out_of_range = {}

    for feature in payload.get("features", []):
        properties = feature["properties"]
        longitude, latitude = feature["geometry"]["coordinates"][:2]

        row = {
            "public_id": public_id,
            "longitude": longitude,
            "latitude": latitude,
            "report_count": properties.get("count", 0),
        }

        for mmi_value, count in (properties.get("count_mmi") or {}).items():
            level = int(mmi_value)
            if level in MMI_LEVELS:
                row[f"mmi_{level}"] = count
            else:
                # The FRR survey offers six cartoons mapping to MMI 3 to 8, so
                # anything outside that range should not exist. A handful of
                # such reports do appear in the archive (single figures across
                # the whole catalogue) and GeoNet's own summary mmi field never
                # uses them. They are dropped, but counted so the caller can see
                # it happened rather than losing data silently.
                out_of_range[level] = out_of_range.get(level, 0) + count

        rows.append(row)

    if out_of_range and warn_out_of_range:
        detail = ", ".join(f"MMI {k}: {v}" for k, v in sorted(out_of_range.items()))
        print(f"[{public_id}] dropped {sum(out_of_range.values())} out-of-scale reports ({detail})")

    frame = pd.DataFrame(rows)

    # An MMI level absent for this event means zero reports at that level,
    # not missing data.
    for level in MMI_LEVELS:
        column = f"mmi_{level}"
        if column not in frame.columns:
            frame[column] = 0

    if not frame.empty:
        columns = [f"mmi_{level}" for level in MMI_LEVELS]
        frame[columns] = frame[columns].fillna(0).astype(int)
        # report_count comes straight from GeoNet and includes any dropped
        # out-of-scale reports, so recompute it from the levels we kept.
        frame["report_count"] = frame[columns].sum(axis=1)

    return frame


def build_dataset(
    min_magnitude=4.0,
    max_magnitude=8.0,
    start_date="2016-10-01T00:00:00",
    end_date="2025-12-31T23:59:59",
    bin_width=0.5,
    max_per_bin=60,
    min_reports=100,
    random_state=7,
    force=False,
    verbose=True,
):
    """Run the whole acquisition and return (events, felt_reports).

    Steps, in order:
      1. Pull the GeoNet catalogue for the magnitude and date window.
      2. Drop teleseismic events and duplicate magnitude solutions.
      3. Sample evenly across magnitude bins.
      4. Fetch felt reports for each sampled event.
      5. Drop events below min_reports.

    Every step is parameterised, so a different study period or magnitude
    range needs no code changes. Defaults reproduce the dataset described in
    the README.

    A note on min_reports: it is deliberately permissive. The real quality
    control happens later at the cell level, where cells with too few reports
    are discarded. Filtering hard here would throw away whole earthquakes
    that still contain a handful of well observed locations, and would strip
    out the smaller magnitudes entirely.
    """
    events = fetch_events(min_magnitude, max_magnitude, start_date, end_date, force=force)
    if verbose:
        print(f"catalogue returned {len(events)} events")

    events = drop_duplicate_events(filter_to_new_zealand(events))
    if verbose:
        print(f"{len(events)} remain after removing teleseismic and duplicate entries")

    events = select_stratified_events(
        events, bin_width=bin_width, max_per_bin=max_per_bin, random_state=random_state
    )
    if verbose:
        print(f"{len(events)} sampled across magnitude bins")

    felt = fetch_felt_reports_for_events(
        events["public_id"], min_reports=min_reports, force=force, verbose=False
    )

    kept_ids = set(felt["public_id"].unique()) if not felt.empty else set()
    events = events[events["public_id"].isin(kept_ids)].reset_index(drop=True)

    if verbose:
        total = int(felt["report_count"].sum()) if not felt.empty else 0
        print(f"{len(events)} events cleared the {min_reports} report threshold")
        print(f"{len(felt)} reporting locations, {total} felt reports in total")

    return events, felt


def fetch_felt_reports_for_events(public_ids, min_reports=0, force=False, verbose=True):
    """Fetch felt reports for many events and return them as one DataFrame.

    Events whose total report count falls below min_reports are dropped, which
    is how sparsely observed earthquakes get excluded.
    """
    frames = []
    for index, public_id in enumerate(public_ids, start=1):
        frame = fetch_felt_reports(public_id, force=force)
        total_reports = int(frame["report_count"].sum()) if not frame.empty else 0

        if total_reports >= min_reports and not frame.empty:
            frames.append(frame)

        if verbose:
            kept = "kept" if total_reports >= min_reports and not frame.empty else "skipped"
            print(f"[{index}/{len(public_ids)}] {public_id}: {total_reports} reports, {kept}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
