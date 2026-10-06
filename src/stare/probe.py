"""Full-resolution probe check of COARSE anchor candidates (run only when the gates fail).

The thumbnail COARSE correlates sees tissue SHAPE (1/65 on a 66k slide), and a shape can look
the same upside-down: the sweep then returns two near-tied angles and neither acceptance gate
passes. Individual nuclei do not repeat under such a flip, so the question "which candidate is
right?" is answered at full resolution, on a handful of small patches:

* **Where.** Up to :data:`PROBE_COUNT` boxes of :data:`PROBE_SIDE` px in the reference frame,
  on the thumbnail cells with the most tissue.
* **What.** For each candidate ``M0`` the moving crop its inverse map draws from is rigid-warped
  into the box (read through ``warp.source_region``, as REG_TILE does), both patches are
  area-averaged to 1/:data:`PROBE_DOWN`, DoG high-passed (nuclei-scale texture) and
  cross-correlated over shifts of up to half the box -- wider than REG_TILE's +-stride capture
  on purpose, and independent of ``--stride``.
* **Verdict.** A probe CONFIRMS a candidate when its correlation peak stands
  >= :data:`PROBE_MIN_RATIO` over the highest value elsewhere, and its shift agrees with the
  median shift of that candidate's other confirming probes (a rigid anchor is off by one
  translation, plus the rotation step over the slide). A candidate is VERIFIED with
  >= :data:`PROBE_MIN_CONFIRMED` confirming probes and strictly more than any other candidate.
  The median shift of the confirming probes is the anchor's measured translation error and is
  returned so the caller can remove it.

One box pair is in memory at a time and the warp runs at the reduced scale, so the stage's
peak RSS is unchanged (measured on an 8k synthetic slide at ``--max-dim 1024``). The thresholds are derived on synthetic nuclei fields
(``tests/test_coarse_probe.py``); they have not been measured on real slides.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["probe_boxes", "probe_shift", "verify_candidates"]

PROBE_SIDE = 1024  # full-res px per probe box
PROBE_COUNT = 8
PROBE_DOWN = (
    4  # correlate at 1/4: nuclei stay several px wide, capture is +-PROBE_SIDE/2
)
PROBE_HIGHPASS_SIGMA = 3.0  # px at the correlated scale (the DoG's wide Gaussian)
PROBE_BLUR_SIGMA = 1.0
PROBE_EXCLUDE_RADIUS = 5  # px at the correlated scale: the peak's own lobe
PROBE_MIN_RATIO = 1.5
PROBE_MIN_CONFIRMED = 3
PROBE_MIN_TISSUE = 0.25  # of a thumbnail cell, to be worth probing
PROBE_SHIFT_TOL_PX = 64.0  # full-res; the caller adds the rotation step over the slide
RATIO_CAP = 1000.0


def probe_boxes(ref_thumb, factor, ref_shape, n=PROBE_COUNT, side=PROBE_SIDE):
    """Up to ``n`` reference-frame boxes ``(x0, y0, x1, y1)`` on the most tissue-rich cells.

    ``ref_thumb`` is the reference nuclear thumbnail (decimated by ``factor``), ``ref_shape``
    the full-resolution ``(H, W)``.
    """
    from stare.coarse_align import _preprocess

    h, w = (int(v) for v in ref_shape)
    side_x, side_y = min(side, w), min(side, h)
    tissue = _preprocess(ref_thumb) > 0
    cell = max(1, int(math.ceil(side / factor)))
    scored = []
    for cy in range(0, tissue.shape[0], cell):
        for cx in range(0, tissue.shape[1], cell):
            frac = float(tissue[cy : cy + cell, cx : cx + cell].mean())
            if frac >= PROBE_MIN_TISSUE:
                scored.append((frac, cy, cx))
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    boxes = []
    for _frac, cy, cx in scored:
        # Centre the box on the cell, clamped inside the slide.
        x0 = int(min(max((cx + cell / 2.0) * factor - side_x / 2.0, 0), w - side_x))
        y0 = int(min(max((cy + cell / 2.0) * factor - side_y / 2.0, 0), h - side_y))
        box = (x0, y0, x0 + side_x, y0 + side_y)
        if box not in boxes:
            boxes.append(box)
        if len(boxes) == n:
            break
    return boxes


def _quarter(patch):
    """Area-average a full-res patch to 1/PROBE_DOWN (float32)."""
    from skimage.transform import downscale_local_mean

    a = np.asarray(patch, dtype=np.float32)
    k = PROBE_DOWN
    a = a[: a.shape[0] // k * k, : a.shape[1] // k * k]
    return downscale_local_mean(a, (k, k)).astype(np.float32)


def _signal(q):
    """DoG high-pass and zero-mean a 1/PROBE_DOWN patch. Returns (array, norm)."""
    from scipy.ndimage import gaussian_filter

    s = gaussian_filter(q, PROBE_BLUR_SIGMA) - gaussian_filter(q, PROBE_HIGHPASS_SIGMA)
    s -= np.float32(s.mean())
    return s, float(np.linalg.norm(s))


def probe_shift(ref_patch, mov_patch):
    """``(peak ratio, (dx, dy) full-res px, peak NCC)`` of one probe box.

    ``(dx, dy)`` is what to ADD to the anchor's translation: the reference content sits at the
    warped moving content shifted by it. A blank patch on either side returns a ratio of 0.
    """
    return _shift(_quarter(ref_patch), _quarter(mov_patch))


def _shift(ref_q, mov_q):
    """:func:`probe_shift` on patches already at 1/PROBE_DOWN."""
    from scipy.fft import irfft2, next_fast_len, rfft2

    a, na = _signal(ref_q)
    b, nb = _signal(mov_q)
    if na <= 0 or nb <= 0:
        return 0.0, (0.0, 0.0), 0.0
    h, w = a.shape
    fft = (next_fast_len(2 * h, real=True), next_fast_len(2 * w, real=True))
    c = irfft2(
        rfft2(a, s=fft, workers=-1) * np.conj(rfft2(b, s=fft, workers=-1)), s=fft
    )
    # shifts in [-h/2, h/2] x [-w/2, w/2]: at least half the box still overlaps
    ry, rx = h // 2, w // 2
    rows = np.r_[0 : ry + 1, fft[0] - ry : fft[0]]
    cols = np.r_[0 : rx + 1, fft[1] - rx : fft[1]]
    sub = c[np.ix_(rows, cols)] / (na * nb)
    i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak = float(sub[i, j])
    if not np.isfinite(peak) or peak <= 0:
        return 0.0, (0.0, 0.0), 0.0
    dy = rows[i] - (fft[0] if rows[i] > ry else 0)
    dx = cols[j] - (fft[1] if cols[j] > rx else 0)
    # the highest value outside the peak's own lobe, on the signed-shift layout
    sy = np.where(rows > ry, rows - fft[0], rows)[:, None]
    sx = np.where(cols > rx, cols - fft[1], cols)[None, :]
    away = (sy - dy) ** 2 + (sx - dx) ** 2 > PROBE_EXCLUDE_RADIUS**2
    second = float(sub[away].max()) if away.any() else 0.0
    ratio = min(peak / second, RATIO_CAP) if second > 0 else RATIO_CAP
    return ratio, (float(dx * PROBE_DOWN), float(dy * PROBE_DOWN)), peak


def _warped_moving(mov_src, index, m0, box):
    """The moving nuclear channel rigid-warped into ``box`` through ``m0``, at 1/PROBE_DOWN,
    or None if the candidate maps the box outside the moving slide.

    The crop is area-averaged FIRST and the warp runs at the reduced scale: a full-resolution
    ``warp.warp_image`` of the box measured ~120 MB of coordinate arrays, this a few MB.
    """
    from stare.coarse_align import _warp_xy
    from stare.warp import source_region

    x0, y0, x1, y1 = box
    k = PROBE_DOWN
    _c, mh, mw = mov_src.shape
    sx0, sy0, sx1, sy1 = source_region(
        m0, None, (x0, y0), (y1 - y0, x1 - x0), src_shape=(mh, mw)
    )
    if sx1 - sx0 < k or sy1 - sy0 < k:
        return None
    crop_q = _quarter(mov_src[index, slice(sy0, sy1), slice(sx0, sx1)])
    # reduced pixel p sits at full-res k p + (k - 1) / 2 (the area average's centre)
    d = np.array([[k, 0, (k - 1) / 2.0], [0, k, (k - 1) / 2.0], [0, 0, 1.0]])
    t_box = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])
    t_crop = np.array([[1, 0, -sx0], [0, 1, -sy0], [0, 0, 1.0]])
    out_to_in = np.linalg.inv(d) @ t_crop @ np.linalg.inv(m0) @ t_box @ d
    return _warp_xy(crop_q, out_to_in, ((y1 - y0) // k, (x1 - x0) // k))


def _consistent(shifts, tol):
    """The largest set of shifts within ``tol`` px of their own median: (indices, median)."""
    idx = list(range(len(shifts)))
    while idx:
        med = np.median(np.array([shifts[i] for i in idx]), axis=0)
        dist = [float(np.hypot(*(np.array(shifts[i]) - med))) for i in idx]
        worst = int(np.argmax(dist))
        if dist[worst] <= tol:
            return idx, med
        idx.pop(worst)
    return [], np.zeros(2)


def verify_candidates(ref_src, mov_src, index, candidates, boxes, shift_tol_px):
    """Score every candidate ``M0`` (3x3, FULL-res, moving -> reference) on every probe box.

    ``ref_src`` / ``mov_src`` are ``(C, H, W)`` array-likes read by region (lazy zarr views or
    arrays). Returns ``(winner, shift, rows)``: the index of the verified candidate or None,
    its measured ``(dx, dy)`` translation error (full-res px, to ADD to its translation), and
    one report row per candidate (``evaluated``, ``confirmed``, ``median_ratio``).
    """
    results = [[] for _ in candidates]  # per candidate: (ratio, shift)
    for box in boxes:
        x0, y0, x1, y1 = box
        ref_q = _quarter(ref_src[index, slice(y0, y1), slice(x0, x1)])
        for k, m0 in enumerate(candidates):
            mov_q = _warped_moving(mov_src, index, np.asarray(m0, float), box)
            if mov_q is None:
                continue
            ratio, shift, _peak = _shift(ref_q, mov_q)
            results[k].append((ratio, shift))
    rows, shifts = [], []
    for res in results:
        good = [s for r, s in res if r >= PROBE_MIN_RATIO]
        keep, med = _consistent(good, shift_tol_px)
        rows.append(
            {
                "evaluated": len(res),
                "confirmed": len(keep),
                "median_ratio": (
                    round(float(np.median([r for r, _s in res])), 3) if res else None
                ),
            }
        )
        shifts.append((float(med[0]), float(med[1])))
    counts = [r["confirmed"] for r in rows]
    if not counts:
        return None, (0.0, 0.0), rows
    best = int(np.argmax(counts))
    need = min(PROBE_MIN_CONFIRMED, max(2, len(boxes)))
    if counts[best] >= need and all(
        counts[best] > c for k, c in enumerate(counts) if k != best
    ):
        return best, shifts[best], rows
    return None, (0.0, 0.0), rows
