"""HyReCo re-stained pairs (IEEE DataPort, doi:10.21227/pzj5-bs61) as benchmark cases.

Subset B is 54 sections stained H&E, scanned, then re-stained PHH3 and scanned again, with
about 43 manually placed landmark pairs each: the same tissue imaged twice, with independent
ground truth. The H&E scan is the reference and the PHH3 scan the moving slide.

Layout expected under ``--data-root`` (override with ``--ref-dir`` / ``--mov-dir``)::

    HE/<case>.tif    HE/<case>.csv
    PHH3/<case>.tif  PHH3/<case>.csv

which is also subset A's layout, so any two of its stain folders can be compared the same way.

The landmark CSVs give one point per line in millimetres from the image's upper-left corner.
They are converted to pixels with the slide's pixel size: ``--pixel-size-um``, else the TIFF's
resolution tags. Landmarks that then fall outside the image stop the case with an error
rather than being scored, because that means the pixel size is wrong.

STARE reads one nuclear channel, so each slide is converted once to inverted luminance (the
``lum`` proxy of ``benchmarks/anhir``); methods that read brightfield get the original TIFF.

NOT YET RUN ON THE REAL FILES: the layout and units above are from the dataset page, and the
TIFF details (axes, pyramid, resolution tags) are handled generally but were not seen.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..cases import Case, case_dir, write_case
from ..imageio import _rgb_view, tiff_mm_per_px, write_nuclear_tiles

TILE = 1024


def read_points_mm(path):
    """``(N, 2)`` x, y from a file of ``x,y[,z]`` lines; a header or blank line is skipped."""
    out = []
    for line in Path(path).read_text().splitlines():
        tok = [t for t in re.split(r"[,;\s]+", line.strip()) if t]
        try:
            out.append((float(tok[0]), float(tok[1])))
        except (IndexError, ValueError):
            continue
    return np.array(out, dtype=float).reshape(-1, 2)


def ensure_proxy(tif_path, out):
    """Inverted luminance of an RGB whole-slide TIFF as a one-channel OME-TIFF; ``(H, W)``."""
    import tifffile

    with tifffile.TiffFile(tif_path) as tif:
        read, (h, w) = _rgb_view(tif.series[0].levels[0])
        if not Path(out).exists():
            def tiles():
                for y0 in range(0, h, TILE):
                    band = read(y0, min(y0 + TILE, h)).astype(np.float32)
                    lum = 0.299 * band[..., 0] + 0.587 * band[..., 1] + 0.114 * band[..., 2]
                    inv = (255.0 - lum).clip(0, 255).astype(np.uint8)
                    for x0 in range(0, w, TILE):
                        yield inv[:, x0 : x0 + TILE]

            write_nuclear_tiles(out, (h, w), np.uint8, tiles(), TILE)
    return h, w


def prepare_one(cases_root, cid, ref_dir, mov_dir, pixel_size_um=None, force=False):
    d = case_dir(cases_root, "hyreco", cid)
    if (d / "case.json").exists() and not force:
        return d
    ref_t, mov_t = Path(ref_dir) / f"{cid}.tif", Path(mov_dir) / f"{cid}.tif"
    cache = Path(cases_root) / "_cache" / "hyreco"
    ref_n = cache / Path(ref_dir).name / f"{cid}.ome.tif"
    mov_n = cache / Path(mov_dir).name / f"{cid}.ome.tif"
    ref_hw, mov_hw = ensure_proxy(ref_t, ref_n), ensure_proxy(mov_t, mov_n)
    pts, px_um = [], None
    for tif, hw in ((ref_t, ref_hw), (mov_t, mov_hw)):
        mm = pixel_size_um / 1000.0 if pixel_size_um else tiff_mm_per_px(tif)
        if not mm:
            raise RuntimeError(f"{tif}: no resolution tags; give --pixel-size-um")
        xy = read_points_mm(tif.with_suffix(".csv")) / mm
        if not len(xy):
            raise RuntimeError(f"{tif.with_suffix('.csv')}: no landmarks read")
        if xy.min() < -1 or xy[:, 0].max() > hw[1] or xy[:, 1].max() > hw[0]:
            raise RuntimeError(
                f"{tif.name}: landmarks span x<={xy[:, 0].max():.0f}, y<={xy[:, 1].max():.0f} px "
                f"on a {hw[1]} x {hw[0]} px image at {mm * 1000:.4f} um/px; wrong pixel size?")
        pts.append(xy)
        px_um = px_um or mm * 1000.0
    n = min(len(pts[0]), len(pts[1]))  # paired by line
    case = Case(
        dataset="hyreco", case_id=str(cid), group=f"{Path(ref_dir).name}-{Path(mov_dir).name}",
        ref_nuclear=str(ref_n), mov_nuclear=str(mov_n), ref_image=str(ref_t),
        mov_image=str(mov_t), modality="brightfield", diagonal=math.hypot(*ref_hw),
        pixel_size_um=px_um, ref_hw=list(ref_hw), mov_hw=list(mov_hw),
        extra={"n_landmarks_ref": len(pts[0]), "n_landmarks_mov": len(pts[1]), "proxy": "lum"},
    )
    return write_case(cases_root, case, pts[1][:n], pts[0][:n])


def _job(job):
    try:
        return job[1], str(prepare_one(*job)), ""
    except Exception as exc:  # one unreadable slide must not stop the others
        return job[1], "", f"{type(exc).__name__}: {exc}"


def prepare(a):
    ref_dir = Path(a.ref_dir or Path(a.data_root) / "HE")
    mov_dir = Path(a.mov_dir or Path(a.data_root) / "PHH3")
    ids = sorted({p.stem for p in ref_dir.glob("*.tif")} & {p.stem for p in mov_dir.glob("*.tif")},
                 key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))
    if not ids:
        raise SystemExit(f"no <case>.tif present in both {ref_dir} and {mov_dir}")
    if a.index is not None:
        ids = ids[a.index : a.index + 1]
    jobs = [(a.cases, cid, ref_dir, mov_dir, a.pixel_size_um, a.force) for cid in ids]
    if a.workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(a.workers, 4)) as pool:
            done = pool.map(_job, jobs)
    else:
        done = [_job(j) for j in jobs]
    for cid, _, err in done:
        if err:
            print(f"hyreco/{cid}: FAILED {err}")
    bad = sum(bool(e) for _, _, e in done)
    print(f"hyreco: {len(done) - bad}/{len(done)} cases prepared under {a.cases}")
    return 1 if bad == len(done) else 0
