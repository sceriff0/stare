"""The full-resolution probe check that settles an anchor the COARSE gates could not accept.

A synthetic nuclei field (blurred random dots under a blobby tissue density) is rotated and
shifted into a "moving" slide. The candidates are the true anchor displaced by a known offset,
its 180-degree flip (the ambiguity a tissue outline cannot resolve) and, separately, an
unrelated field.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tifffile
from scipy.ndimage import affine_transform, gaussian_filter
from stare import coarse_align as ca
from stare import probe
from stare.stages import coarse as coarse_stage

N = 2048


def _field(seed, n=N):
    r = np.random.default_rng(seed)
    dens = gaussian_filter(r.random((n, n)).astype(np.float32), n / 12)
    dens = (dens - dens.min()) / np.ptp(dens)
    dots = (r.random((n, n)) < 0.004 * dens).astype(np.float32)
    return (gaussian_filter(dots, 5) * 4e6 + 300 * dens).astype(np.float32)


def _m(theta, tx, ty, n=N):
    t = math.radians(theta)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    c = np.array([(n - 1) / 2.0] * 2)
    m = np.eye(3)
    m[:2, :2] = r
    m[:2, 2] = c - r @ c + [tx, ty]
    return m


def _moving(ref, m_true, seed=7):
    """``mov(q) = ref(m_true q)``, with a gain change and noise."""
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    mov = affine_transform(
        ref, swap @ m_true[:2, :2] @ swap, offset=m_true[:2, 2][::-1], order=1
    )
    noise = np.random.default_rng(seed).standard_normal(mov.shape)
    return np.clip(mov * 0.7 + 20 * noise, 0, None).astype(np.float32)


@pytest.fixture(scope="module")
def pair():
    ref = _field(1)
    m_true = _m(-99.5, 40, -25)
    return ref, _moving(ref, m_true), m_true


def test_the_true_candidate_is_verified_and_its_offset_measured(pair):
    ref, mov, m_true = pair
    off = m_true.copy()
    off[:2, 2] += [-60, 44]  # the anchor is (+60, -44) px short of the truth
    flip = _m(-99.5 + 180, 40, -25)
    boxes = probe.probe_boxes(ref[::8, ::8], 8, ref.shape)
    assert len(boxes) >= 3, boxes
    winner, shift, rows = probe.verify_candidates(
        ref[None], mov[None], 0, [flip, off], boxes, 64.0
    )
    assert winner == 1, rows
    assert shift == pytest.approx((60, -44), abs=4), shift
    assert rows[0]["confirmed"] == 0 and rows[1]["confirmed"] == len(boxes), rows


def test_no_candidate_is_verified_on_an_unrelated_slide(pair):
    ref, _mov, m_true = pair
    boxes = probe.probe_boxes(ref[::8, ::8], 8, ref.shape)
    winner, _shift, rows = probe.verify_candidates(
        ref[None], _field(2)[None], 0, [m_true, _m(80.5, 40, -25)], boxes, 64.0
    )
    assert winner is None and all(r["confirmed"] == 0 for r in rows), rows


def test_a_candidate_mapped_off_the_moving_slide_is_not_evaluated(pair):
    ref, mov, m_true = pair
    away = m_true.copy()
    away[:2, 2] += 10 * N
    boxes = probe.probe_boxes(ref[::8, ::8], 8, ref.shape)
    winner, _shift, rows = probe.verify_candidates(
        ref[None], mov[None], 0, [away, m_true], boxes, 64.0
    )
    assert rows[0]["evaluated"] == 0 and winner == 1, rows


def _run_stage(tmp_path, ref, mov, extra=()):
    ref_f, mov_f = tmp_path / "ref.ome.tiff", tmp_path / "mov.ome.tiff"
    for f, img in ((ref_f, ref), (mov_f, mov)):
        tifffile.imwrite(
            str(f),
            img[None].astype(np.uint16),
            photometric="minisblack",
            tile=(256, 256),
        )
    out = tmp_path / "m0.json"
    coarse_stage.main(
        [
            "--reference",
            str(ref_f),
            "--moving",
            str(mov_f),
            "--max-dim",
            "256",
            "--out-m0",
            str(out),
            "--out-tiles",
            str(tmp_path / "tiles.csv"),
            *extra,
        ]
    )
    return json.loads(out.read_text())


def _gates_shut(monkeypatch):
    """Sweep never trusted, no ORB fit: the anchor reaches the stage untrusted."""
    monkeypatch.setattr(ca, "MIN_PEAK_RATIO", 1e9)
    monkeypatch.setattr(
        ca, "_orb_fallback", lambda r, m, model: (None, float("nan"), 0)
    )


def test_the_stage_trusts_an_anchor_the_probes_confirm(
    tmp_path, monkeypatch, pair, caplog
):
    ref, mov, m_true = pair
    _gates_shut(monkeypatch)
    with caplog.at_level("WARNING"):
        doc = _run_stage(tmp_path, ref, mov)
    assert doc["coarse_trusted"] is True and doc["coarse_probe"]["verified"] is True, (
        doc
    )
    assert "CONFIRMED" in caplog.text
    m0 = np.asarray(doc["M0"])
    c = np.array([N / 2.0, N / 2.0, 1.0])
    assert np.linalg.norm((m0 @ c - m_true @ c)[:2]) <= 8.0, (m0, m_true)


def test_gates_that_pass_run_no_probe(tmp_path, pair):
    ref, mov, _m_true = pair
    doc = _run_stage(tmp_path, ref, mov)
    assert doc["coarse_trusted"] is True and doc["coarse_probe"] is None, doc


def test_an_unconfirmed_anchor_warns_by_default_and_fails_under_strict(
    tmp_path, monkeypatch, pair, caplog
):
    ref, _mov, _m_true = pair
    _gates_shut(monkeypatch)
    other = _field(2)
    with caplog.at_level("WARNING"):
        doc = _run_stage(tmp_path, ref, other)
    assert doc["coarse_trusted"] is False and doc["coarse_probe"]["verified"] is False
    assert "UNVERIFIED ANCHOR" in caplog.text and "could not confirm" in caplog.text
    (tmp_path / "m0.json").unlink()
    with pytest.raises(ca.CoarseRefused, match="strict-anchor"):
        _run_stage(tmp_path, ref, other, extra=("--strict-anchor",))
    assert not (tmp_path / "m0.json").exists()
