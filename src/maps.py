"""Render a shake map for every shortlisted model, as image files.

src/spatial.py answers whether a model attenuates properly as a number. This
module answers the same question as a picture.

A shake map is what an emergency response team would actually look at: for one
earthquake, how strongly does the model say each part of the country shook.
The pictures are not the test. Looking at maps is exactly what the original
version of this project did, and it is unreliable, because two models with
very different attenuation scores can look similar at a glance and a model can
misbehave at a magnitude nobody thought to plot. The numeric check in
spatial.py is the test. These maps exist to show what a verdict means, and to
make the shape of a failure legible once the number has already caught it.

Every map is drawn for the same earthquake on the same colour scale, so the
panels can be compared directly rather than each being read on its own terms.

Run it directly to regenerate everything into maps/:

    python src/maps.py

Outputs:

    maps/contact-sheet.png          all fourteen models together
    maps/selected-vs-rejected.png   the two that matter, side by side
    maps/models/NN_<model>.png      one map per model, in accuracy order

Predictions are masked to land using assets/nz_coastline.json, a simplified
Natural Earth coastline committed with the repository so the maps render
offline.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench as B
import models as M
import spatial as S
from clean import apply_weight_scheme, expand_to_weighted_labels
from features import MODEL_FEATURES, model_features

# Everything is resolved against the repository root rather than the working
# directory, so `python src/maps.py` behaves the same from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
COASTLINE_PATH = REPO_ROOT / "assets" / "nz_coastline.json"

# The largest earthquake in the catalogue, and the one whose shaking pattern a
# New Zealand reader will recognise. A large event is the right one to draw: it
# is where the models saw the least training data and where they differ most.
KAIKOURA = {
    "name": "Kaikoura, 14 November 2016",
    "magnitude": 7.8,
    "depth_km": 15.0,
    "longitude": 173.02,
    "latitude": -42.69,
}

# The full range the Felt RAPID Report survey can express, shared across every
# panel. Fixing the scale is what makes the maps comparable, and it is how
# published shake maps are drawn for the same reason.
#
# It also carries a finding. Most of these maps stay pale even at the epicentre
# of a magnitude 7.8 earthquake, because most of the candidates never predict
# strong shaking anywhere. Only the two Hist Gradient Boosting classifiers
# reach the top of the scale. Contours are drawn every half unit so the shape
# of each decay stays readable regardless of how much colour the model uses.
COLOUR_RANGE = (3.0, 8.0)
COLOUR_MAP = "YlOrRd"
CONTOUR_LEVELS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]

PASS_COLOUR = "#1b7837"
FAIL_COLOUR = "#b2182b"


def slugify(name):
    """A filename that survives every operating system."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def load_coastline(path=COASTLINE_PATH):
    """New Zealand's coastline as a list of longitude/latitude rings."""
    with open(path, encoding="utf-8") as handle:
        return [np.asarray(ring) for ring in json.load(handle)["polygons"]]


def land_mask(longitude, latitude, rings):
    """True where a point falls on land.

    Uses matplotlib's point-in-polygon rather than shapely so that drawing the
    maps needs nothing beyond what the rest of the project already imports.
    """
    points = np.column_stack([np.ravel(longitude), np.ravel(latitude)])
    inside = np.zeros(len(points), dtype=bool)
    for ring in rings:
        inside |= MplPath(ring).contains_points(points)
    return inside.reshape(np.shape(longitude))


def grid_shape(grid):
    """The (rows, columns) layout behind nz_grid's flattened frame."""
    return grid["northing"].nunique(), grid["easting"].nunique()


def predict_field(candidate, event, grid, rings, feature_columns=None):
    """Predicted intensity over the grid, as a 2D array with sea set to NaN."""
    feature_columns = MODEL_FEATURES if feature_columns is None else feature_columns
    mapped = S.shake_map(candidate.estimator, feature_columns, event,
                         grid=grid, kind=candidate.kind)
    rows, columns = grid_shape(grid)

    longitude = mapped["longitude"].to_numpy().reshape(rows, columns)
    latitude = mapped["latitude"].to_numpy().reshape(rows, columns)
    intensity = mapped["predicted_mmi"].to_numpy().reshape(rows, columns)

    return longitude, latitude, np.where(land_mask(longitude, latitude, rings),
                                         intensity, np.nan)


def draw(ax, field, event, rings, vmin=COLOUR_RANGE[0], vmax=COLOUR_RANGE[1]):
    """Draw one predicted intensity field onto an axis."""
    longitude, latitude, intensity = field

    mesh = ax.pcolormesh(longitude, latitude, intensity, cmap=COLOUR_MAP,
                         vmin=vmin, vmax=vmax, shading="nearest")

    # Contours make the shape of the decay readable. A model that attenuates
    # properly draws closed rings around the epicentre; a model that does not
    # draws bands, islands, or nothing at all.
    with np.errstate(invalid="ignore"):
        ax.contour(longitude, latitude, intensity, levels=CONTOUR_LEVELS,
                   colors="#444444", linewidths=0.4, alpha=0.7)

    for ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#333333", linewidth=0.4)

    ax.plot(event["longitude"], event["latitude"], marker="*", markersize=15,
            color="#08306b", markeredgecolor="white", markeredgewidth=0.8,
            zorder=5)

    ax.set_xlim(166, 179)
    ax.set_ylim(-47.5, -34.2)
    ax.set_aspect(1 / np.cos(np.radians(abs(event["latitude"]))))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return mesh


def title_for(row, rank=None):
    """A panel title carrying the verdict and the number behind it."""
    verdict = "passes" if row["passes"] else "rejected"
    prefix = f"{rank}. " if rank is not None else ""
    return (f"{prefix}{row['model']}\n"
            f"MAE {row['cell_mae']:.4f}   Spearman {row['worst_spearman']:.3f}   {verdict}")


def contact_sheet(fields, physical, event, rings, path, columns=5):
    """Every model on one page, ordered by accuracy.

    Read left to right the maps get less accurate. Read by title colour, only
    two of them are usable. The two orderings do not agree, which is the
    finding this project exists to make.
    """
    rows = int(np.ceil(len(physical) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 3.5 * rows),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for position, (_, row) in enumerate(physical.iterrows()):
        mesh = draw(axes[position], fields[row["model"]], event, rings)
        axes[position].set_title(title_for(row, rank=position + 1), fontsize=7,
                                 color=PASS_COLOUR if row["passes"] else FAIL_COLOUR)

    for ax in axes[len(physical):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Predicted intensity for {event['name']}, magnitude {event['magnitude']}\n"
        "Ordered by accuracy. Green passed the physical check, red was rejected.",
        fontsize=12)
    fig.colorbar(mesh, ax=axes.tolist(), label="Predicted MMI",
                 fraction=0.02, pad=0.01)

    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    return path


def comparison(fields, physical, event, rings, path):
    """The selected model beside the more accurate one that was rejected."""
    best_overall = physical.iloc[0]
    best_passing = physical[physical["passes"]].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), layout="constrained")
    for ax, row, label in [(axes[0], best_passing, "SELECTED"),
                           (axes[1], best_overall, "REJECTED")]:
        mesh = draw(ax, fields[row["model"]], event, rings)
        ax.set_title(f"{label}: {row['model']}\n"
                     f"cell MAE {row['cell_mae']:.4f},  worst Spearman {row['worst_spearman']:.3f},"
                     f"  drop {row['total_drop']:.2f} MMI",
                     fontsize=10, color=PASS_COLOUR if row["passes"] else FAIL_COLOUR)

    fig.suptitle(f"{event['name']}, magnitude {event['magnitude']}\n"
                 "The rejected model is the more accurate of the two.", fontsize=12)
    fig.colorbar(mesh, ax=axes.tolist(), label="Predicted MMI",
                 fraction=0.03, pad=0.01)

    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def individual_maps(fields, physical, event, rings, directory):
    """One map per model, named so the folder reads in accuracy order."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    for position, (_, row) in enumerate(physical.iterrows(), start=1):
        fig, ax = plt.subplots(figsize=(5, 6), layout="constrained")
        mesh = draw(ax, fields[row["model"]], event, rings)
        ax.set_title(title_for(row), fontsize=10,
                     color=PASS_COLOUR if row["passes"] else FAIL_COLOUR)
        fig.colorbar(mesh, ax=ax, label="Predicted MMI", fraction=0.04)

        path = directory / f"{position:02d}_{slugify(row['model'])}.png"
        fig.savefig(path, dpi=120, facecolor="white")
        plt.close(fig)
        written.append(path)

    return written


def prepare(features_path=None, bench_path=None, shortlist_size=14, verbose=True):
    """Load the data, refit the shortlist, and run the physical check.

    Returns the fitted candidates, the same verdict table notebook 04 builds,
    and the feature list actually used, so the maps and the numbers can never
    disagree.
    """
    features_path = features_path or REPO_ROOT / "data" / "processed" / "features.csv"
    bench_path = bench_path or REPO_ROOT / "data" / "processed" / "bench_results.csv"

    features = pd.read_csv(features_path)
    columns = model_features(features, verbose=verbose)

    labels = apply_weight_scheme(
        expand_to_weighted_labels(features, feature_columns=columns + ["mmi_mean"]),
        scheme="sqrt")
    labels["is_weekend"] = labels["is_weekend"].astype(int)
    train, _ = M.split_by_event(labels, test_size=0.2, verbose=False)

    results = pd.read_csv(bench_path)
    shortlist = (results[~results["status"].str.startswith("failed")]
                 .nsmallest(shortlist_size, "cell_mae"))

    if verbose:
        print(f"refitting {len(shortlist)} candidates on {len(train):,} rows")
    fitted = B.fit_shortlist(shortlist["model"], train, columns)

    checks = []
    for name, candidate in fitted.items():
        summary = S.physical_plausibility(candidate.estimator, columns,
                                          kind=candidate.kind)
        verdict = S.plausibility_verdict(summary)
        checks.append({"model": name,
                       "worst_spearman": round(summary["worst_spearman"], 3),
                       "total_drop": round(summary["mean_total_drop"], 2),
                       "passes": verdict["passes"]})

    physical = (pd.DataFrame(checks)
                .merge(shortlist[["model", "cell_mae"]], on="model")
                .sort_values("cell_mae")
                .reset_index(drop=True))

    return fitted, physical, columns


def render_all(output_dir=None, event=None, spacing_km=5, verbose=True):
    """Regenerate every map. This is what running the module does."""
    event = event or KAIKOURA
    output_dir = Path(output_dir) if output_dir else REPO_ROOT / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    fitted, physical, columns = prepare(verbose=verbose)
    rings = load_coastline()
    grid = S.nz_grid(spacing_km=spacing_km)

    fields = {name: predict_field(candidate, event, grid, rings, columns)
              for name, candidate in fitted.items()}

    if verbose:
        on_land = int(np.isfinite(next(iter(fields.values()))[2]).sum())
        print(f"{len(grid):,} grid points, {on_land:,} on land, "
              f"{int(physical['passes'].sum())} of {len(physical)} models pass")
        strongest = max(np.nanmax(field[2]) for field in fields.values())
        print(f"strongest prediction on any map: MMI {strongest:.2f} "
              f"(colour scale tops out at {COLOUR_RANGE[1]})")

    written = [contact_sheet(fields, physical, event, rings, output_dir / "contact-sheet.png"),
               comparison(fields, physical, event, rings, output_dir / "selected-vs-rejected.png")]
    written += individual_maps(fields, physical, event, rings, output_dir / "models")

    if verbose:
        for path in written:
            print(f"  wrote {path.name}")

    return written


if __name__ == "__main__":
    render_all()
