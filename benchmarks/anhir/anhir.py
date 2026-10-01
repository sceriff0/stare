"""Score STARE on the ANHIR challenge (Borovec et al. 2020) with the challenge's own metrics.

ANHIR scores *landmarks*, never pixels: each case's source landmarks are moved into the
target image's frame, and the error is the distance to the target landmarks divided by the
image diagonal (rTRE). STARE registers the source (moving) onto the target (reference), and
its manifest is a forward map moving -> reference, so the warped source landmarks are
``stare.stage_warp.make_warper(manifest)("source", xy, stage)`` -- no inversion involved.

Subcommands, each idempotent:

``join``   rejoin the split image archive (``dataset_medium.z01..z05`` + ``.zip``) and unzip
           it to ``<data-root>/images`` (needs Info-ZIP ``zip``/``unzip``).
``list``   print the number of selected cases (to size a SLURM array) or their ids.
``run``    register ONE case (``--case-id``, or ``--task-index`` into the selected list, which
           is what a SLURM array task passes) and write its warped landmarks.
``score``  score every method found under ``--out`` against the target landmarks.

ANHIR images are brightfield RGB JPEGs; STARE reads one nuclear channel. ``run`` converts
each image once (cached, atomic) to a one-channel tiled OME-TIFF holding a nuclear proxy:
``lum`` = inverted luminance (dark nuclei -> bright, background -> 0, as DAPI) or ``hema`` =
the haematoxylin channel of Ruifrok-Johnston colour deconvolution (``skimage.color.rgb2hed``).

Only ``training`` cases have public target landmarks; ``evaluation`` cases are scored by the
challenge server. Report training numbers as training numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COVER = "dataset_medium.csv"
STATUSES = ("training", "evaluation")
PROXIES = ("lum", "hema")
REF_NAME, MOV_NAME = "target", "source"  # slide names inside the STARE manifest
STARE_METHODS = ("stare_rigid", "stare")  # manifest stages "rigid" and "refined"
BAND_ROWS = 1024  # rows per band when building the proxy, to bound memory


# ── cover table ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Case:
    case_id: int
    tissue: str
    status: str
    source_image: str
    target_image: str
    source_landmarks: str
    target_landmarks: str
    diagonal: float


def tissue_of(rel):
    """``COAD_01/scale-25pc/S1.jpg`` -> ``COAD``; ``lung-lesion_1/...`` -> ``lung-lesion``."""
    return rel.split("/", 1)[0].rsplit("_", 1)[0]


def load_cases(data_root):
    with open(Path(data_root) / COVER, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cases = []
    for r in rows:
        cases.append(
            Case(
                case_id=int(r[""]),
                tissue=tissue_of(r["Source image"]),
                status=r["status"].strip(),
                source_image=r["Source image"],
                target_image=r["Target image"],
                source_landmarks=r["Source landmarks"],
                target_landmarks=r["Target landmarks"],
                # the challenge's own diagonal, never recomputed
                diagonal=float(r["Image diagonal [pixels]"]),
            )
        )
    return cases


def select(cases, status="training", tissues=None, case_ids=None, limit=None):
    out = [
        c
        for c in cases
        if (status in (None, "all") or c.status == status)
        and (not tissues or c.tissue in tissues)
        and (not case_ids or c.case_id in case_ids)
    ]
    out.sort(key=lambda c: c.case_id)
    return out[:limit] if limit else out


# ── landmarks ─────────────────────────────────────────────────────────────────
def read_landmarks(path):
    """``(N, 2)`` X, Y (= column, row) from an ANHIR ``,X,Y`` CSV."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([[float(r["X"]), float(r["Y"])] for r in rows], dtype=float).reshape(-1, 2)


def write_landmarks(path, xy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["", "X", "Y"])
        for i, (x, y) in enumerate(np.asarray(xy, dtype=float).reshape(-1, 2)):
            w.writerow([i, repr(float(x)), repr(float(y))])
    os.replace(tmp, path)
    return path


def read_imagej_points(path):
    """The bUnwarpJ baseline's ``point\\nN\\nx y ...`` format."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines or lines[0].lower() != "point":
        raise ValueError(f"{path}: not an ImageJ point file")
    n = int(lines[1])
    return np.array([[float(v) for v in ln.split()[:2]] for ln in lines[2 : 2 + n]]).reshape(-1, 2)


def source_landmarks_path(case, data_root):
    """The official archive if unpacked, else the copy the bUnwarpJ baseline folder carries."""
    official = Path(data_root) / "landmarks" / case.source_landmarks
    if official.exists():
        return official, "landmarks"
    fallback = Path(data_root) / "BmUnwarpJ" / str(case.case_id) / "source_landmarks.csv"
    if fallback.exists():
        return fallback, "BmUnwarpJ"
    raise FileNotFoundError(
        f"no source landmarks for case {case.case_id}: unpack the ANHIR landmark archive "
        f"to {Path(data_root) / 'landmarks'}"
    )


# ── metrics (https://anhir.grand-challenge.org/Performance_Metrics/) ──────────
def case_stats(warped, target, source, diagonal):
    n = min(len(warped), len(target), len(source))
    tre = np.hypot(*(warped[:n] - target[:n]).T)
    r = tre / diagonal
    r_init = np.hypot(*(source[:n] - target[:n]).T) / diagonal
    return {
        "n_landmarks": n,
        "rtre_median": float(np.median(r)),
        "rtre_mean": float(np.mean(r)),
        "rtre_max": float(np.max(r)),
        "tre_median_px": float(np.median(tre)),
        # fraction of landmarks the registration brought closer than the initial pose
        "robustness": float(np.mean(r < r_init)),
    }


# ── join ──────────────────────────────────────────────────────────────────────
def cmd_join(a):
    root = Path(a.data_root)
    last, joined, images = root / "dataset_medium.zip", root / "dataset_medium_joined.zip", root / "images"
    if not joined.exists():
        subprocess.run(["zip", "-s", "0", str(last), "--out", str(joined)], check=True)
    images.mkdir(exist_ok=True)
    cmd = ["unzip", "-q", "-n", str(joined)]
    if a.only:
        cmd += a.only
    subprocess.run(cmd + ["-d", str(images)], check=True)
    print(f"images -> {images}")
    return 0


# ── proxy conversion ──────────────────────────────────────────────────────────
def _find_image(images_root, rel):
    """The cover table's path, tolerating one extra top-level folder from the archive."""
    p = Path(images_root) / rel
    if p.exists():
        return p
    hits = list(Path(images_root).glob(f"*/{rel}"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{p} missing -- run `anhir.py join` first")


def proxy_image(jpeg, proxy):
    """Nuclear proxy as a uint8 (H, W) array: nuclei bright, background dark."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(jpeg) as im:
        rgb = np.asarray(im.convert("RGB"))
    if proxy == "lum":
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        return (255.0 - lum).clip(0, 255).astype(np.uint8)
    from skimage.color import rgb2hed

    h = np.empty(rgb.shape[:2], dtype=np.float32)
    for y0 in range(0, rgb.shape[0], BAND_ROWS):
        h[y0 : y0 + BAND_ROWS] = rgb2hed(rgb[y0 : y0 + BAND_ROWS])[..., 0]
    lo, hi = np.percentile(h[::8, ::8], (1.0, 99.9))
    return ((h - lo) / max(hi - lo, 1e-6) * 255.0).clip(0, 255).astype(np.uint8)


def ensure_proxy(data_root, work, rel, proxy):
    """Convert one image once; atomic, so concurrent array tasks can share the cache."""
    import tifffile

    out = Path(work) / "images" / proxy / (str(Path(rel).with_suffix("")) + ".ome.tif")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = proxy_image(_find_image(Path(data_root) / "images", rel), proxy)
    fd, tmp = tempfile.mkstemp(suffix=".ome.tif", dir=out.parent)
    os.close(fd)
    tifffile.imwrite(
        tmp,
        arr[None],
        ome=True,
        tile=(512, 512),
        compression="zlib",
        metadata={"axes": "CYX", "Channel": {"Name": ["NUCLEAR"]},
                  "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0},
    )
    os.replace(tmp, out)
    return out


# ── run one case ──────────────────────────────────────────────────────────────
def register_case(ref, mov, manifest, tre, workdir, a):
    """`stare register`'s own code path; STITCH is skipped unless --stitch (ANHIR scores points)."""
    import stare.stages.stitch as stitch_stage
    from stare.cli import build_parser, register

    argv = [
        "register", "--reference", str(ref), "--moving", str(mov),
        "--out", str(Path(workdir) / "registered.ome.tif"), "--manifest", str(manifest),
        "--tre", str(tre), "--workdir", str(workdir), "--workers", str(a.workers),
        "--reference-name", REF_NAME, "--moving-name", MOV_NAME,
        "--tile", str(a.tile), "--halo", str(a.halo), "--stride", str(a.stride),
        "--max-dim", str(a.max_dim), "--model", a.model,
    ]
    ns = build_parser().parse_args(argv)
    if a.stitch:
        return register(ns)
    real = stitch_stage.main
    stitch_stage.main = lambda argv: 0
    try:
        return register(ns)
    finally:
        stitch_stage.main = real


def cmd_run(a):
    from stare.stage_warp import make_warper

    cases = select(load_cases(a.data_root), a.status, a.tissue,
                   None if a.case_id is None else [a.case_id])
    if a.case_id is None:
        if a.task_index is None:
            raise SystemExit("give --case-id or --task-index")
        if a.task_index >= len(cases):
            print(f"task {a.task_index}: only {len(cases)} cases selected; nothing to do")
            return 0
        case = cases[a.task_index]
    else:
        if not cases:
            raise SystemExit(f"case {a.case_id} not in the selection")
        case = cases[0]

    out = Path(a.out)
    run_dir = out / "runs" / str(case.case_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"case_id": case.case_id, "tissue": case.tissue, "status": case.status,
           "proxy": a.proxy, "params": {k: getattr(a, k) for k in
                                         ("tile", "halo", "stride", "max_dim", "model", "workers")},
           "ok": False, "error": ""}
    t0 = time.perf_counter()
    try:
        src_path, rec["source_landmarks_from"] = source_landmarks_path(case, a.data_root)
        ref = ensure_proxy(a.data_root, a.work, case.target_image, a.proxy)
        mov = ensure_proxy(a.data_root, a.work, case.source_image, a.proxy)
        rec["convert_s"] = time.perf_counter() - t0
        manifest, tre = run_dir / "manifest.json", run_dir / "tre.json"
        t1 = time.perf_counter()
        rc = register_case(ref, mov, manifest, tre, run_dir / "work", a)
        rec["register_s"] = time.perf_counter() - t1
        if rc:
            raise RuntimeError(f"stare register exited {rc}")
        warp = make_warper(json.loads(manifest.read_text()))
        source = read_landmarks(src_path)
        write_landmarks(out / "stare_rigid" / f"{case.case_id}.csv", warp(MOV_NAME, source, "rigid"))
        write_landmarks(out / "stare" / f"{case.case_id}.csv", warp(MOV_NAME, source, "refined"))
        rec["ok"] = True
        if tre.exists():
            rec["solve"] = json.loads(tre.read_text()).get("solve", {})
    except Exception as exc:  # a failed case is scored at the initial pose, as the challenge does
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        print(f"case {case.case_id}: {rec['error']}", file=sys.stderr)
    finally:
        rec["total_s"] = time.perf_counter() - t0
        if not a.keep_work:
            import shutil

            shutil.rmtree(run_dir / "work", ignore_errors=True)
        (run_dir / "run.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"case {case.case_id} ({case.tissue}): {'ok' if rec['ok'] else 'FAILED'} "
          f"in {rec['total_s']:.0f} s")
    return 0 if rec["ok"] else 1


# ── score ─────────────────────────────────────────────────────────────────────
def _method_points(method, case, a):
    """Warped source landmarks for one (method, case), or None if the method produced none."""
    if method == "initial":
        return read_landmarks(Path(a.data_root) / "landmarks" / case.source_landmarks)
    if method == "bunwarpj":
        p = Path(a.data_root) / "BmUnwarpJ" / str(case.case_id) / "warped_source_landmarks.txt"
        return read_imagej_points(p) if p.exists() else None
    p = Path(a.out) / method / f"{case.case_id}.csv"
    return read_landmarks(p) if p.exists() else None


def _time_min(method, case, a):
    if method == "bunwarpj":
        p = Path(a.data_root) / "BmUnwarpJ" / str(case.case_id) / "TIME.txt"
        return float(p.read_text().strip()) / 60000.0 if p.exists() else math.nan
    if method in STARE_METHODS:
        p = Path(a.out) / "runs" / str(case.case_id) / "run.json"
        if p.exists():
            return json.loads(p.read_text()).get("register_s", math.nan) / 60.0
    return math.nan


def _agg(rows):
    med = np.array([r["rtre_median"] for r in rows])
    return {
        "n_cases": len(rows),
        "n_imputed": sum(r["imputed_initial"] for r in rows),
        "avg_median_rtre": float(np.mean(med)),
        "med_median_rtre": float(np.median(med)),
        "avg_max_rtre": float(np.mean([r["rtre_max"] for r in rows])),
        "avg_robustness": float(np.mean([r["robustness"] for r in rows])),
        "avg_rank": float(np.mean([r["rank"] for r in rows])),
        "median_time_min": float(np.nanmedian([r["time_min"] for r in rows]))
        if any(not math.isnan(r["time_min"]) for r in rows) else math.nan,
    }


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_score(a):
    lroot = Path(a.data_root) / "landmarks"
    if not lroot.exists():
        raise SystemExit(f"no landmark archive at {lroot}; scoring needs the target landmarks")
    cases = select(load_cases(a.data_root), "training", a.tissue)
    methods = ["initial"]
    if (Path(a.data_root) / "BmUnwarpJ").exists():
        methods.append("bunwarpj")
    methods += [m for m in STARE_METHODS if (Path(a.out) / m).exists()]

    rows = []
    for c in cases:
        src_p, tgt_p = lroot / c.source_landmarks, lroot / c.target_landmarks
        if not (src_p.exists() and tgt_p.exists()):
            continue
        source, target = read_landmarks(src_p), read_landmarks(tgt_p)
        per_case = []
        for m in methods:
            pts = _method_points(m, c, a)
            imputed = pts is None
            stats = case_stats(source if imputed else pts, target, source, c.diagonal)
            per_case.append({"case_id": c.case_id, "tissue": c.tissue, "method": m, **stats,
                             "time_min": _time_min(m, c, a), "imputed_initial": imputed})
        order = np.argsort(np.argsort([r["rtre_median"] for r in per_case], kind="stable"))
        for r, k in zip(per_case, order):
            r["rank"] = int(k) + 1
        rows += per_case
    if not rows:
        raise SystemExit("no training case had both landmark files")

    tables = Path(a.out) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    _write_csv(tables / "cases.csv", rows)
    # "common": the cases every method actually registered. bUnwarpJ's published baseline has
    # output for only 84 of the 230 training cases; on "all" its gaps count at the initial pose
    # (the challenge rule), so compare methods head to head on "common".
    imputed = {r["case_id"] for r in rows if r["imputed_initial"] and r["method"] != "initial"}
    agg = []
    for m in methods:
        mine = [r for r in rows if r["method"] == m]
        agg.append({"method": m, "subset": "all", **_agg(mine)})
        common = [r for r in mine if r["case_id"] not in imputed]
        if common:
            agg.append({"method": m, "subset": "common", **_agg(common)})
        for t in sorted({r["tissue"] for r in mine}):
            agg.append({"method": m, "subset": f"tissue:{t}", **_agg([r for r in mine if r["tissue"] == t])})
    _write_csv(tables / "aggregates.csv", agg)

    print(f"{len(rows) // len(methods)} training cases scored; tables -> {tables}\n")
    hdr = f"{'method':<12}{'subset':<8}{'cases':>6}{'imputed':>9}{'avg medRTRE':>13}{'med medRTRE':>13}{'avg robust':>12}{'avg rank':>10}{'time min':>10}"
    print(hdr)
    for r in agg:
        if r["subset"] in ("all", "common"):
            print(f"{r['method']:<12}{r['subset']:<8}{r['n_cases']:>6}{r['n_imputed']:>9}{r['avg_median_rtre']:>13.5f}"
                  f"{r['med_median_rtre']:>13.5f}{r['avg_robustness']:>12.3f}{r['avg_rank']:>10.2f}"
                  f"{r['median_time_min']:>10.2f}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data-root", required=True, help="folder holding dataset_medium.csv")
        p.add_argument("--status", default="training", choices=("all",) + STATUSES)
        p.add_argument("--tissue", nargs="*", default=None)

    j = sub.add_parser("join")
    j.add_argument("--data-root", required=True)
    j.add_argument("--only", nargs="*", default=None, help="extract only these archive members")

    lst = sub.add_parser("list")
    common(lst)
    lst.add_argument("--ids", action="store_true", help="print ids, not the count")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--work", required=True, help="cache for the converted OME-TIFFs")
    r.add_argument("--out", required=True)
    r.add_argument("--case-id", type=int, default=None)
    r.add_argument("--task-index", type=int, default=None,
                   help="index into the selected cases (SLURM_ARRAY_TASK_ID)")
    r.add_argument("--proxy", default="lum", choices=PROXIES)
    r.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    r.add_argument("--tile", type=int, default=2048)
    r.add_argument("--halo", type=int, default=256)
    r.add_argument("--stride", type=int, default=128)
    r.add_argument("--max-dim", type=int, default=1024)
    r.add_argument("--model", default="euclidean", choices=("euclidean", "similarity", "affine"))
    r.add_argument("--stitch", action="store_true", help="also write the registered image (QC)")
    r.add_argument("--keep-work", action="store_true", help="keep M0, tile plan and control JSONs")

    s = sub.add_parser("score")
    s.add_argument("--data-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tissue", nargs="*", default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "join":
        return cmd_join(a)
    if a.cmd == "list":
        cases = select(load_cases(a.data_root), a.status, a.tissue)
        print("\n".join(str(c.case_id) for c in cases) if a.ids else len(cases))
        return 0
    if a.cmd == "run":
        return cmd_run(a)
    return cmd_score(a)


if __name__ == "__main__":
    raise SystemExit(main())
