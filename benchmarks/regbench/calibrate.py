"""Choose each competitor's options on development cases, before the test cases are scored.

Every candidate below is run on the ``dev_*`` cases (``slurm/submit_calibrate.sh``), each
under its own label. ``regbench calibrate`` then scores them and writes the winner per method
to ``params.lock.json``: the candidate, and the variant of it, with the lowest mean per-case
median rTRE, a failed case counting at its initial pose. ``slurm/submit.sh --arm recommended``
runs the locked options under the label ``<method>_rec``.

The budget is the same for every method: ``BUDGET`` candidates, fixed here before any run.
STARE has none: it runs at its released defaults in both arms, so nothing about it is chosen
on data the benchmark holds. If no calibration was run, ``FALLBACK`` (each package's own
documented higher-accuracy setting) is what the recommended arm uses.

The dev windows are 4096 px, where ``micro_fraction=0.25`` still means VALIS's 4096 px
default, so the first two VALIS candidates coincide there; they differ on larger slides.
"""

from __future__ import annotations

import json
from pathlib import Path

BUDGET = 6
CANDIDATES = {
    "valis": [
        {},
        {"micro_fraction": "0.25"},
        {"micro_fraction": "1.0"},
        {"micro_fraction": "0.25", "micro_rigid": "1"},
        {"micro_fraction": "0.25", "max_processed_image_dim_px": "1024"},
        {"micro_fraction": "0.25", "max_non_rigid_registration_dim_px": "4096"},
    ],
    "deeperhistreg": [
        {"config": c, "direction": d}
        for c in ("default_initial_nonrigid", "default_initial_nonrigid_high_resolution",
                  "default_initial_nonrigid_fast")
        for d in ("swapped", "native")
    ],
}
FALLBACK = {
    "valis": {"micro_fraction": "0.25"},
    "deeperhistreg": {"config": "default_initial_nonrigid_high_resolution"},
}
assert all(len(v) == BUDGET for v in CANDIDATES.values())


def label(method, i):
    return f"{method}_c{i}"


def configs():
    """``[(method, label, opts)]`` of every candidate."""
    return [(m, label(m, i), o) for m, cands in CANDIDATES.items() for i, o in enumerate(cands)]


def recommended(method, lock=None):
    """``(opts, source)`` of a method's recommended arm."""
    if lock and Path(lock).exists():
        rec = json.loads(Path(lock).read_text()).get(method)
        if rec:
            return rec["opts"], f"calibrated ({rec['label']})"
    if method in FALLBACK:
        return FALLBACK[method], "documented setting (no calibration found)"
    return {}, "defaults"


def choose(rows):
    """The lock: per method the best (candidate, variant) of the scored dev landmark rows."""
    import numpy as np

    # where real-image dev cases exist, choose on the headline tier only
    if any(r.get("tier") == "headline" for r in rows):
        rows = [r for r in rows if r.get("tier") == "headline"]
    lock = {}
    for method, cands in CANDIDATES.items():
        table = []
        for i, opts in enumerate(cands):
            lab = label(method, i)
            for variant in sorted({r["method"] for r in rows
                                   if r["method"] == lab or r["method"].startswith(lab + "_")}):
                mine = [r for r in rows if r["method"] == variant]
                table.append({"label": lab, "variant": variant, "opts": opts,
                              "n_cases": len(mine),
                              "n_failed": int(sum(bool(r["imputed_initial"]) for r in mine)),
                              "mean_median_rtre": float(np.mean([r["rtre_median"] for r in mine])),
                              "median_tre_px": float(np.median([r["tre_px_median"] for r in mine]))})
        if not table:
            continue
        # ties go to the earlier candidate, i.e. towards the package default
        best = min(table, key=lambda t: t["mean_median_rtre"])
        lock[method] = {**{k: best[k] for k in ("label", "variant", "opts", "mean_median_rtre")},
                        "candidates": table}
    return lock


def main(a):
    if a.print_configs:
        for m, lab, opts in configs():
            print(m, lab, " ".join(f"{k}={v}" for k, v in opts.items()))
        return 0
    from .cases import write_json
    from .score import score

    rows = score(a.cases, a.out, dev=True)["landmarks"]
    lock = choose(rows)
    if not lock:
        raise SystemExit(f"no calibration runs under {a.out}; run slurm/submit_calibrate.sh first")
    path = Path(a.lock or Path(a.out) / "params.lock.json")
    write_json(path, lock)
    for m, rec in lock.items():
        print(f"{m}: {rec['label']} -> {rec['variant']}  "
              f"{' '.join(f'{k}={v}' for k, v in rec['opts'].items()) or '(defaults)'}  "
              f"mean median rTRE {rec['mean_median_rtre']:.3g}")
    print(f"lock -> {path}")
    return 0
