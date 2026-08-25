"""Tests for the event filtering logic in src/ingest.py.

These use synthetic frames rather than live API responses, so they run
offline and stay fast. The real coordinates used are taken from actual
GeoNet catalogue entries returned by a November 2016 query, which is where
the teleseismic and duplicate problems were first noticed.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ingest
from ingest import (assign_magnitude_bins, drop_duplicate_events,
                    filter_to_new_zealand, select_stratified_events, summarise_bins)


def make_events(rows):
    return pd.DataFrame(
        rows,
        columns=["public_id", "origin_time", "magnitude", "depth_km", "longitude", "latitude"],
    )


def test_keeps_mainland_new_zealand_events():
    events = make_events([
        # Kaikoura mainshock
        ["2016p858000", "2016-11-13T11:02:56Z", 7.82, 15.1, 173.022141, -42.692535],
    ])
    assert len(filter_to_new_zealand(events)) == 1


def test_drops_teleseismic_events():
    events = make_events([
        ["nz", "2016-11-13T11:02:56Z", 7.82, 15.1, 173.022141, -42.692535],
        # Chile
        ["cl", "2016-11-20T20:57:43Z", 6.37, 115.0, -68.764, -31.643],
        # Japan
        ["jp", "2016-11-21T20:59:49Z", 6.69, 9.0, 141.387, 37.393],
    ])
    kept = filter_to_new_zealand(events)
    assert list(kept["public_id"]) == ["nz"]


def test_keeps_chatham_islands_across_the_antimeridian():
    """The Chathams sit just past 180 degrees, so a naive 165-180 box drops them."""
    events = make_events([
        ["chatham", "2020-01-01T00:00:00Z", 5.0, 20.0, -176.5, -43.9],
    ])
    assert len(filter_to_new_zealand(events)) == 1


def test_drops_duplicate_magnitude_solutions_keeping_largest():
    """GeoNet publishes several entries for one event, differing only in magnitude."""
    events = make_events([
        ["jp_a", "2016-11-21T20:59:49Z", 6.686873, 9.0, 141.387, 37.393],
        ["jp_b", "2016-11-21T20:59:49Z", 6.910880, 9.0, 141.387, 37.393],
        ["jp_c", "2016-11-21T20:59:49Z", 6.811184, 9.0, 141.387, 37.393],
    ])
    deduped = drop_duplicate_events(events)
    assert len(deduped) == 1
    assert deduped.iloc[0]["public_id"] == "jp_b"


def test_distinct_events_at_the_same_location_are_kept():
    """Aftershocks share a location but differ in time, so both must survive."""
    events = make_events([
        ["a", "2016-11-13T11:02:56Z", 7.82, 15.1, 173.022141, -42.692535],
        ["b", "2016-11-13T11:05:14Z", 6.02, 4.9, 173.022141, -42.692535],
    ])
    assert len(drop_duplicate_events(events)) == 2


@pytest.mark.parametrize("frame", [pd.DataFrame(), make_events([])])
def test_empty_input_is_handled(frame):
    assert filter_to_new_zealand(frame).empty
    assert drop_duplicate_events(frame).empty


def spread_of_events():
    magnitudes = [4.0, 4.2, 4.4, 4.6, 4.9, 5.0, 5.3, 5.6, 6.1, 6.4, 7.0, 7.9]
    return make_events([
        [f"e{i}", f"2020-01-{i + 1:02d}T00:00:00Z", m, 10.0, 173.0, -42.0]
        for i, m in enumerate(magnitudes)
    ])


def test_magnitude_bins_are_left_closed():
    """A magnitude of exactly 5.0 belongs to 5.0-5.5, not 4.5-5.0."""
    events = make_events([
        ["a", "2020-01-01T00:00:00Z", 4.999, 10.0, 173.0, -42.0],
        ["b", "2020-01-02T00:00:00Z", 5.000, 10.0, 173.0, -42.0],
    ])
    bins = assign_magnitude_bins(events)["magnitude_bin"].tolist()
    assert bins == ["4.5-5.0", "5.0-5.5"]


def test_stratification_caps_each_bin():
    selected = select_stratified_events(spread_of_events(), max_per_bin=2)
    assert (summarise_bins(selected)["events"] <= 2).all()


def test_stratification_keeps_the_bin_label():
    """Regression test: groupby.apply silently dropped the grouping column."""
    selected = select_stratified_events(spread_of_events(), max_per_bin=2)
    assert "magnitude_bin" in selected.columns
    assert selected["magnitude_bin"].notna().all()


def test_sparse_bins_contribute_everything_they_have():
    """A bin holding one event should still contribute it, not be skipped."""
    selected = select_stratified_events(spread_of_events(), max_per_bin=2)
    counts = summarise_bins(selected).set_index("magnitude_bin")["events"]
    assert counts["7.5-8.0"] == 1


def test_selection_is_reproducible():
    first = select_stratified_events(spread_of_events(), max_per_bin=2, random_state=7)
    second = select_stratified_events(spread_of_events(), max_per_bin=2, random_state=7)
    assert first["public_id"].tolist() == second["public_id"].tolist()


def test_max_per_bin_none_keeps_everything():
    events = spread_of_events()
    selected = select_stratified_events(events, max_per_bin=None)
    assert len(selected) == len(events)
    assert "magnitude_bin" in selected.columns


def fake_intensity_payload(points):
    """Build an API-shaped payload. Each point is (lon, lat, {mmi: count})."""
    return {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "count": sum(counts.values()),
                    "mmi": max(counts, key=lambda k: int(k)) if counts else None,
                    "count_mmi": counts,
                },
            }
            for lon, lat, counts in points
        ]
    }


@pytest.fixture
def stub_api(monkeypatch):
    """Replace the network call so felt report parsing can be tested offline."""
    def install(payload):
        monkeypatch.setattr(ingest, "_get_json", lambda *a, **k: payload)
    return install


def test_felt_reports_expand_the_mmi_distribution(stub_api):
    stub_api(fake_intensity_payload([(174.0, -41.0, {"3": 2, "5": 1})]))
    frame = ingest.fetch_felt_reports("test")

    assert frame.loc[0, "mmi_3"] == 2
    assert frame.loc[0, "mmi_5"] == 1
    assert frame.loc[0, "mmi_4"] == 0
    assert frame.loc[0, "report_count"] == 3


def test_all_mmi_levels_exist_even_when_absent(stub_api):
    """A level with no reports is a zero, not a missing value."""
    stub_api(fake_intensity_payload([(174.0, -41.0, {"3": 1})]))
    frame = ingest.fetch_felt_reports("test")

    expected = [f"mmi_{level}" for level in ingest.MMI_LEVELS]
    assert all(column in frame.columns for column in expected)
    assert not frame[expected].isna().any().any()


def test_out_of_scale_reports_are_dropped(stub_api):
    """The FRR survey only produces MMI 3 to 8, but the archive holds a few strays."""
    stub_api(fake_intensity_payload([(174.0, -41.0, {"1": 2, "2": 1, "4": 5})]))
    frame = ingest.fetch_felt_reports("test", warn_out_of_range=False)

    assert "mmi_1" not in frame.columns
    assert "mmi_2" not in frame.columns
    assert frame.loc[0, "mmi_4"] == 5
    # report_count is recomputed from kept levels, not taken from the API
    assert frame.loc[0, "report_count"] == 5


def test_report_count_always_reconciles(stub_api):
    stub_api(fake_intensity_payload([
        (174.0, -41.0, {"3": 4, "6": 2}),
        (175.0, -40.0, {"1": 9, "8": 1}),
    ]))
    frame = ingest.fetch_felt_reports("test", warn_out_of_range=False)

    levels = [f"mmi_{level}" for level in ingest.MMI_LEVELS]
    assert (frame[levels].sum(axis=1) == frame["report_count"]).all()
