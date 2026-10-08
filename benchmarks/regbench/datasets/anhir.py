"""ANHIR training cases as benchmark cases.

Everything ANHIR-specific (the cover table, the landmark CSVs, the brightfield -> nuclear
proxy) is ``benchmarks/anhir/anhir.py``; this only lays its cases out in the shared contract.
The source image is the moving slide and the target image the reference, so the source
landmarks are the points to warp and the target landmarks the truth.

A method that reads brightfield gets the original JPEGs (``ref_image`` / ``mov_image``); a
method that needs one nuclear channel gets the proxy OME-TIFFs. Both are the same pixel grid,
so the landmarks apply to either.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ..cases import Case, case_dir, write_case
from ..imageio import shape_of


def harness():
    """``benchmarks/anhir/anhir.py`` as a module (it is a script beside this package)."""
    path = Path(__file__).resolve().parents[2] / "anhir" / "anhir.py"
    if "anhir_harness" in sys.modules:
        return sys.modules["anhir_harness"]
    spec = importlib.util.spec_from_file_location("anhir_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["anhir_harness"] = mod  # its dataclass looks its own module up by name
    spec.loader.exec_module(mod)
    return mod


def prepare_one(cases_root, data_root, c, proxy="lum", force=False, h=None):
    h = h or harness()
    d = case_dir(cases_root, "anhir", c.case_id)
    if (d / "case.json").exists() and not force:
        return d
    lroot = Path(data_root) / "landmarks"
    source = h.read_landmarks(lroot / c.source_landmarks)
    target = h.read_landmarks(lroot / c.target_landmarks)
    n = min(len(source), len(target))  # paired by index, as the challenge scores them
    cache = Path(cases_root) / "_cache" / "anhir"
    images = Path(data_root) / "images"
    ref_n = h.ensure_proxy(data_root, cache, c.target_image, proxy)
    mov_n = h.ensure_proxy(data_root, cache, c.source_image, proxy)
    case = Case(
        dataset="anhir", case_id=str(c.case_id), group=c.tissue,
        ref_nuclear=str(ref_n), mov_nuclear=str(mov_n),
        ref_hw=list(shape_of(ref_n)[1:]), mov_hw=list(shape_of(mov_n)[1:]),
        ref_image=str(h._find_image(images, c.target_image)),
        mov_image=str(h._find_image(images, c.source_image)),
        modality="brightfield", diagonal=c.diagonal,
        extra={"proxy": proxy, "source_image": c.source_image, "target_image": c.target_image},
    )
    return write_case(cases_root, case, source[:n], target[:n])


def _job(job):
    cases_root, data_root, c, proxy, force = job
    try:
        return c.case_id, str(prepare_one(cases_root, data_root, c, proxy, force)), ""
    except Exception as exc:  # one unreadable case must not stop the other 229
        return c.case_id, "", f"{type(exc).__name__}: {exc}"


def prepare(a):
    h = harness()
    lroot = Path(a.data_root) / "landmarks"
    if not lroot.exists():
        raise SystemExit(f"no landmark archive at {lroot}; see benchmarks/anhir/README.md")
    cases = h.select(h.load_cases(a.data_root), "training", a.tissue, a.case_id, a.limit)
    cases = [c for c in cases
             if (lroot / c.source_landmarks).exists() and (lroot / c.target_landmarks).exists()]
    if a.index is not None:
        cases = cases[a.index : a.index + 1]
    jobs = [(a.cases, a.data_root, c, a.proxy, a.force) for c in cases]
    if a.workers > 1 and len(jobs) > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
            done = pool.map(_job, jobs)
    else:
        done = [_job(j) for j in jobs]
    bad = [(cid, err) for cid, _, err in done if err]
    for cid, err in bad:
        print(f"anhir/{cid}: FAILED {err}")
    print(f"anhir: {len(done) - len(bad)}/{len(done)} cases prepared under {a.cases}")
    return 1 if done and len(bad) == len(done) else 0
