"""Score every variant found under ``--out`` against the cases; needs no method's software.

Two families of numbers, by what a case carries:

**Landmarks** (synthetic, ANHIR) -- the ANHIR challenge's metrics
(https://anhir.grand-challenge.org/Performance_Metrics/): TRE is the distance between a warped
moving landmark and its reference landmark, rTRE that distance over the reference image's
diagonal; per case the median / mean / max rTRE, the robustness (fraction of landmarks brought
closer than the unregistered pose) and the method's rank by median rTRE.

**Cells** (synthetic, multiplex) -- mirage's reg_qc=2 full-transform record: matched Dice,
per-pair IoU and centroid displacement of natively segmented nuclei (``regbench.cells``).

A case a variant failed on is scored at the initial pose (``imputed_initial``), as the
challenge does; the ``common`` subset is the cases every variant actually registered.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import DATASETS, cells
from .cases import list_cases, load_points, polys, result_dir

RESERVED = ("tables", "logs")
FLAT_FIELD_S = 1e5  # STARE's own bound for "no local structure left" (CV_RING_MAX_LOG10_S)
CELL_KEYS = ("n_ref", "n_moving", "n_pairs", "pair_fraction", "pair_fraction_moving",
             "dice_matched", "iou_mean", "iou_p50", "frac_iou_ge_0.5", "displacement_px_p50",
             "displacement_px_p90", "displacement_px_mean", "displacement_um_p50",
             "displacement_um_p90", "match_radius_px")


def landmark_stats(warped, target, source, diagonal):
    """The challenge's per-case numbers. A non-finite warped point counts at its initial position."""
    warped = np.array(warped, dtype=float)
    bad = ~np.isfinite(warped).all(axis=1)
    warped[bad] = source[bad]
    tre = np.hypot(*(warped - target).T)
    init = np.hypot(*(source - target).T)
    r = tre / diagonal
    return {
        "n_landmarks": len(tre), "n_nonfinite": int(bad.sum()),
        "rtre_median": float(np.median(r)), "rtre_mean": float(np.mean(r)),
        "rtre_max": float(np.max(r)),
        "tre_px_median": float(np.median(tre)), "tre_px_mean": float(np.mean(tre)),
        "tre_px_p90": float(np.percentile(tre, 90)), "tre_px_max": float(np.max(tre)),
        "initial_tre_px_median": float(np.median(init)),
        "robustness": float(np.mean(tre < init)),
    }


def resource_row(run, megapixels):
    """One run's cost. ``megapixels`` is both slides together: what the method had to read."""
    r = run.get("resources") or {}
    ok = bool(run.get("ok"))
    wall, cpu = r.get("wall_s", math.nan), r.get("cpu_s", math.nan)
    n = run.get("n_cpus") or math.nan
    budget = min(run.get("workers") or n, n)  # the cores it was told to use

    def gb(key):
        return r[key] / 1024.0 if r.get(key) is not None else math.nan

    return {
        "ok": ok, "megapixels": megapixels, "n_cpus": n, "workers": run.get("workers", math.nan),
        "wall_min": wall / 60.0, "cpu_min": cpu / 60.0,
        # mean cores actually busy, and that as a share of the cores it was given
        "cores_used": cpu / wall if wall else math.nan,
        "cpu_efficiency": cpu / (wall * budget) if wall and budget else math.nan,
        "peak_rss_gb": gb("peak_rss_mb"), "cgroup_peak_gb": gb("cgroup_peak_mb"),
        "gpu_peak_gb": gb("gpu_peak_mb"),
        "wall_s_per_mpx": wall / megapixels if megapixels else math.nan,
        "cpu_s_per_mpx": cpu / megapixels if megapixels else math.nan,
        "warp_s": run.get("warp_s", math.nan), "rss_method": r.get("rss_method", ""),
        "host": run.get("host", ""), "cpu_model": run.get("cpu_model", ""),
        "slurm_job_id": run.get("slurm_job_id") or "", "slurm_task_id": run.get("slurm_task_id") or "",
    }


def _max(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmax(v)) if np.isfinite(v).any() else math.nan


def aggregate_resources(rows):
    """Per (dataset, method), over the runs that succeeded: a failed run's cost is not a cost
    of registering, and is counted in ``n_failed`` instead."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        for m in dict.fromkeys(r["method"] for r in rows if r["dataset"] == ds):
            mine = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            ok = [r for r in mine if r["ok"]]
            if not ok:
                out.append({"dataset": ds, "method": m, "n_runs": 0, "n_failed": len(mine)})
                continue
            cpu = np.array([r["cpu_min"] for r in ok], dtype=float)
            out.append({
                "dataset": ds, "method": m, "n_runs": len(ok), "n_failed": len(mine) - len(ok),
                "n_cpus": _median(ok, "n_cpus"),
                "median_wall_min": _median(ok, "wall_min"), "max_wall_min": _max(ok, "wall_min"),
                "median_cpu_min": _median(ok, "cpu_min"),
                "total_cpu_hours": float(np.nansum(cpu)) / 60.0,
                "median_cores_used": _median(ok, "cores_used"),
                "median_peak_rss_gb": _median(ok, "peak_rss_gb"),
                "max_peak_rss_gb": _max(ok, "peak_rss_gb"),
                "max_cgroup_peak_gb": _max(ok, "cgroup_peak_gb"),
                "max_gpu_peak_gb": _max(ok, "gpu_peak_gb"),
                "median_wall_s_per_mpx": _median(ok, "wall_s_per_mpx"),
                "median_cpu_s_per_mpx": _median(ok, "cpu_s_per_mpx"),
                "rss_method": "/".join(sorted({r["rss_method"] for r in ok})),
                "cpu_models": "; ".join(sorted({r["cpu_model"] for r in ok if r["cpu_model"]})),
            })
    keys = list(dict.fromkeys(k for r in out for k in r))
    return [{k: r.get(k, math.nan) for k in keys} for r in out]


def scaling_table(lm, ce, res):
    """One row per (size, method) of the synthetic ``scale`` suite: accuracy and cost."""
    key = lambda r: (r["case_id"], r["method"])  # noqa: E731
    lm_by = {key(r): r for r in lm if r["group"] == "scale"}
    ce_by = {key(r): r for r in ce if r["group"] == "scale"}
    res_by = {key(r): r for r in res if r["group"] == "scale"}
    rows = []
    for k in sorted(set(lm_by) | set(res_by)):
        a, c, r = lm_by.get(k, {}), ce_by.get(k, {}), res_by.get(k, {})
        mpx = r.get("megapixels", math.nan)
        rows.append({
            "case_id": k[0], "size_px": int(k[0].rsplit("_", 1)[1]), "megapixels": mpx,
            "method": k[1], "status": ("ok" if not a.get("imputed_initial") else
                                       (a.get("problem") or "failed")[:60]),
            "tre_px_median": a.get("tre_px_median", math.nan),
            "tre_px_p90": a.get("tre_px_p90", math.nan),
            "dice_matched": c.get("dice_matched", math.nan),
            "displacement_px_p50": c.get("displacement_px_p50", math.nan),
            "wall_min": r.get("wall_min", math.nan) if r.get("ok") else math.nan,
            "cpu_min": r.get("cpu_min", math.nan) if r.get("ok") else math.nan,
            "peak_rss_gb": r.get("peak_rss_gb", math.nan) if r.get("ok") else math.nan,
            "cgroup_peak_gb": r.get("cgroup_peak_gb", math.nan) if r.get("ok") else math.nan,
            "gpu_peak_gb": r.get("gpu_peak_gb", math.nan) if r.get("ok") else math.nan,
            "wall_s_per_mpx": r.get("wall_s_per_mpx", math.nan) if r.get("ok") else math.nan,
            "stare_flat_field": a.get("stare_flat_field", ""),
        })
    return [r for r in rows if r["method"] != "initial"] and rows


def find_variants(out, dataset):
    out = Path(out)
    found = [d.name for d in sorted(out.iterdir())
             if d.is_dir() and d.name not in RESERVED and d.name != "initial"
             and (d / dataset).is_dir()] if out.exists() else []
    return ["initial", *found]


def _score_case(job):
    out, case, variants = job
    points = load_points(case)
    src = points.get("landmarks_mov")
    tgt = points.get("landmarks_ref")
    ref_cells, mov_cells = polys(points, "cells_ref"), polys(points, "cells_mov")
    lm_rows, cell_rows, res_rows = [], [], []
    mpx = (sum(h * w for h, w in (case.ref_hw, case.mov_hw)) / 1e6
           if case.ref_hw and case.mov_hw else math.nan)
    for v in variants:
        head = {"dataset": case.dataset, "case_id": case.case_id, "group": case.group, "method": v}
        warped, run, problem = None, {}, ""
        if v != "initial":
            d = result_dir(out, v, case)
            if (d / "run.json").exists():
                run = json.loads((d / "run.json").read_text())
            if not run:
                problem = "not run"
            elif not run.get("ok") or not (d / "warped.npz").exists():
                problem = run.get("error") or "no output"
            else:
                with np.load(d / "warped.npz") as z:
                    warped = {k: z[k] for k in z.files}
        imputed = v != "initial" and warped is None
        if run:
            res_rows.append({**head, **resource_row(run, mpx)})
        solve = (run.get("info") or {}).get("solve") or {}
        s_chosen = solve.get("smoothing_s", math.nan)
        tail = {"time_min": run.get("register_s", math.nan) / 60.0 if run.get("ok") else math.nan,
                "imputed_initial": imputed, "problem": problem[:200],
                # STARE only: the smoothing SOLVE chose, and whether it is a flat field (the
                # mesh then adds nothing to the rigid anchor)
                "stare_smoothing_s": s_chosen,
                "stare_flat_field": bool(s_chosen >= FLAT_FIELD_S) if s_chosen == s_chosen else ""}
        if tgt is not None and len(tgt):
            w = src if warped is None else warped["landmarks"]
            lm_rows.append({**head, **landmark_stats(w, tgt, src, case.diagonal), **tail})
        if ref_cells is not None and mov_cells is not None and len(ref_cells) and len(mov_cells):
            moved = mov_cells if warped is None else mov_cells.with_xy(warped["cells_xy"])
            rec = cells.score_cells(ref_cells, moved, case.pixel_size_um)
            cell_rows.append({**head, **{k: rec.get(k, math.nan) for k in CELL_KEYS}, **tail})
    if lm_rows:
        for r, k in zip(lm_rows, rankdata([r["rtre_median"] for r in lm_rows], method="average")):
            r["rank"] = float(k)
    return lm_rows, cell_rows, res_rows


def _mean(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan


def _median(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else math.nan


def _agg_landmarks(rows):
    return {"avg_median_rtre": _mean(rows, "rtre_median"),
            "med_median_rtre": _median(rows, "rtre_median"),
            "avg_max_rtre": _mean(rows, "rtre_max"),
            "med_median_tre_px": _median(rows, "tre_px_median"),
            "avg_robustness": _mean(rows, "robustness"), "avg_rank": _mean(rows, "rank")}


def _agg_cells(rows):
    return {"avg_dice_matched": _mean(rows, "dice_matched"),
            "med_dice_matched": _median(rows, "dice_matched"),
            "avg_pair_fraction": _mean(rows, "pair_fraction"),
            "med_displacement_px_p50": _median(rows, "displacement_px_p50"),
            "med_displacement_px_p90": _median(rows, "displacement_px_p90"),
            "med_displacement_um_p50": _median(rows, "displacement_um_p50")}


def aggregate(rows, agg):
    """One row per (dataset, method, subset): ``all``, ``common`` and each ``group:<name>``."""
    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        mine = [r for r in rows if r["dataset"] == ds]
        imputed = {r["case_id"] for r in mine if r["imputed_initial"]}
        for m in dict.fromkeys(r["method"] for r in mine):
            of_m = [r for r in mine if r["method"] == m]
            subsets = [("all", of_m), ("common", [r for r in of_m if r["case_id"] not in imputed])]
            subsets += [(f"group:{g}", [r for r in of_m if r["group"] == g])
                        for g in sorted({r["group"] for r in of_m})]
            for name, sub in subsets:
                if sub:
                    out.append({"dataset": ds, "method": m, "subset": name, "n_cases": len(sub),
                                "n_imputed": sum(r["imputed_initial"] for r in sub), **agg(sub),
                                "median_time_min": _median(sub, "time_min")})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4g}"
    return str(v)


def markdown(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def score(cases_root, out, datasets=None, workers=1):
    jobs = []
    for ds in datasets or DATASETS:
        cs = list_cases(cases_root, ds)
        if cs:
            variants = find_variants(out, ds)
            jobs += [(str(out), c, variants) for c in cs]
    if not jobs:
        raise SystemExit(f"no prepared cases under {cases_root}")
    if workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            done = pool.map(_score_case, jobs)
    else:
        done = [_score_case(j) for j in jobs]
    lm = [r for rows in done for r in rows[0]]
    ce = [r for rows in done for r in rows[1]]
    res = [r for rows in done for r in rows[2]]
    res_agg = aggregate_resources(res) if res else []
    lm_agg, ce_agg = aggregate(lm, _agg_landmarks), aggregate(ce, _agg_cells)

    tables = Path(out) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(tables / "landmarks_cases.csv", lm)
    write_csv(tables / "landmarks_aggregates.csv", lm_agg)
    write_csv(tables / "cells_cases.csv", ce)
    write_csv(tables / "cells_aggregates.csv", ce_agg)
    write_csv(tables / "resources_cases.csv", res)
    write_csv(tables / "resources_aggregates.csv", res_agg)
    scaling = scaling_table(lm, ce, res)
    write_csv(tables / "scaling.csv", scaling)

    head = [r for r in lm_agg if r["subset"] in ("all", "common")]
    cell_head = [r for r in ce_agg if r["subset"] in ("all", "common")]
    parts = ["# Registration benchmark\n"]
    if head:
        parts += ["## Landmarks (rTRE = TRE / reference diagonal)\n",
                  markdown(head, ["dataset", "method", "subset", "n_cases", "n_imputed",
                                  "avg_median_rtre", "med_median_rtre", "med_median_tre_px",
                                  "avg_robustness", "avg_rank", "median_time_min"]), ""]
    if cell_head:
        parts += ["## Cells (matched Dice and centroid displacement of native nuclei)\n",
                  markdown(cell_head, ["dataset", "method", "subset", "n_cases", "n_imputed",
                                       "avg_dice_matched", "avg_pair_fraction",
                                       "med_displacement_px_p50", "med_displacement_px_p90",
                                       "med_displacement_um_p50", "median_time_min"]), ""]
    if res_agg:
        parts += ["## Resources (whole process tree; successful runs only)\n",
                  markdown([r for r in res_agg if r["n_runs"]],
                           ["dataset", "method", "n_runs", "n_failed", "n_cpus", "median_wall_min",
                            "median_cpu_min", "total_cpu_hours", "median_cores_used",
                            "median_peak_rss_gb", "max_peak_rss_gb", "max_gpu_peak_gb",
                            "median_wall_s_per_mpx", "median_cpu_s_per_mpx"]), ""]
    if scaling:
        parts += ["## Scaling (synthetic `scale` suite: one deformation, growing slide)\n",
                  markdown(scaling, ["size_px", "method", "status", "tre_px_median", "dice_matched",
                                     "wall_min", "cpu_min", "peak_rss_gb", "wall_s_per_mpx"]), ""]
    text = "\n".join(parts)
    (tables / "summary.md").write_text(text)
    print(text)
    print(f"tables -> {tables}")
    return {"landmarks": lm, "cells": ce, "resources": res, "landmarks_agg": lm_agg,
            "cells_agg": ce_agg, "resources_agg": res_agg}
