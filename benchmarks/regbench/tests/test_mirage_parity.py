"""``regbench.cells`` gives the numbers mirage's own reg_qc=2 scorer gives.

The benchmark restates mirage's pairing and Dice so it imports nothing from the pipeline;
this runs the original (``bin/utils/cell_pairs.py``) on the same polygons and compares every
number. Skipped where no mirage checkout is found: set ``MIRAGE_REPO``, or keep mirage at
``../../pipelines/mirage`` relative to this repository.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from regbench import cells

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def cp():
    repo = Path(os.environ.get("MIRAGE_REPO") or ROOT.parents[1] / "pipelines" / "mirage")
    path = repo / "bin" / "utils" / "cell_pairs.py"
    if not path.exists():
        pytest.skip(f"no mirage checkout at {repo}")
    spec = importlib.util.spec_from_file_location("mirage_cell_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mirage_cell_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


def _blob(rng, cx, cy, r):
    t = np.sort(rng.uniform(0, 2 * np.pi, 24))
    rad = r * rng.uniform(0.8, 1.2, 24)
    return np.stack([cx + rad * np.cos(t), cy + rad * np.sin(t)], axis=1)


def _geojson(ps):
    feats = []
    for k in range(len(ps)):
        ring = ps.xy[ps.off[k] : ps.off[k + 1]].tolist()
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}})
    return {"type": "FeatureCollection", "features": feats}


@pytest.mark.parametrize("jitter", [0.0, 1.5, 6.0])
def test_pairs_iou_dice_and_displacement_equal_mirages(cp, jitter):
    rng = np.random.default_rng(3)
    centres = rng.uniform(50, 1500, size=(600, 2))  # dense enough for contested pairings
    ref = cells.from_rings([_blob(rng, x, y, 7) for x, y in centres])
    keep = rng.random(600) > 0.15  # some cells have no partner
    mov = cells.from_rings([_blob(rng, x + dx, y + dy, 7) for (x, y), (dx, dy) in
                            zip(centres[keep], rng.normal(0, jitter, (int(keep.sum()), 2)))])

    mine = cells.score_cells(ref, mov, pixel_size_um=0.325)

    m_ref, m_mov = cp.from_feature_collection(_geojson(ref)), cp.from_feature_collection(_geojson(mov))
    area_r, cent_r = cp.feature_area_centroid(m_ref)
    area_m, cent_m = cp.feature_area_centroid(m_mov)
    radius = 1.5 * cp.median_equivalent_radius(np.concatenate([area_r, area_m]))
    ia, ib, _, _ = cp.match_cells(cent_r, cent_m, radius, method="lsa")
    iou, scored = cp.pair_iou(m_ref, m_mov, ia, ib, supersample=2)
    dist = cp.centroid_distance(cent_r, cent_m, ia, ib)
    theirs = cp.summarize_stage(iou, scored, dist, area_r[ia], area_m[ib], iou_thresh=0.5,
                                pixel_size_um=0.325)

    assert mine["match_radius_px"] == pytest.approx(radius)
    assert mine["n_pairs"] == theirs["n_pairs"] == len(ia)
    assert mine["n_pairs_scored"] == theirs["n_pairs_scored"]
    for k in ("dice_matched", "iou_mean", "iou_p50", "frac_iou_ge_0.5", "displacement_px_p50",
              "displacement_px_p90", "displacement_px_mean", "displacement_um_p50"):
        assert mine[k] == pytest.approx(theirs[k], rel=1e-9, abs=1e-12), k
