"""Tests for the ANHIR harness: metric definitions, landmark I/O and axis order, warp direction,
and the scorer's imputation rule. Run with ``PYTHONPATH=src pytest benchmarks/anhir``."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import anhir  # noqa: E402

COVER_HEADER = [
    "", "Image diagonal [pixels]", "Image size [pixels]", "Source image", "Source landmarks",
    "Target image", "Target landmarks", "status", "Warped target landmarks",
    "Warped source landmarks", "Execution time [minutes]",
]


def _cover(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / anhir.COVER, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COVER_HEADER)
        for cid, status, src, tgt in rows:
            w.writerow([cid, 1000.0, "(800, 600)", f"T_1/scale-25pc/{src}.jpg",
                        f"T_1/scale-25pc/{src}.csv", f"T_1/scale-25pc/{tgt}.jpg",
                        f"T_1/scale-25pc/{tgt}.csv", status, "", "", ""])


def test_case_stats_definitions():
    target = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    source = target + [30.0, 40.0]  # initial TRE 50 px each
    warped = target + [[3.0, 4.0], [0.0, 0.0], [60.0, 80.0]]  # 5, 0, 100 px
    s = anhir.case_stats(warped, target, source, diagonal=1000.0)
    assert s["n_landmarks"] == 3
    assert s["tre_median_px"] == pytest.approx(5.0)
    assert s["rtre_median"] == pytest.approx(0.005)
    assert s["rtre_max"] == pytest.approx(0.1)
    assert s["robustness"] == pytest.approx(2 / 3)  # the 100 px point got worse than 50 px


def test_landmarks_roundtrip_keeps_x_as_column(tmp_path):
    xy = np.array([[12.5, 3.0], [7.0, 99.25]])
    p = anhir.write_landmarks(tmp_path / "a.csv", xy)
    assert p.read_text().splitlines()[0] == ",X,Y"
    np.testing.assert_array_equal(anhir.read_landmarks(p), xy)


def test_manifest_warp_is_moving_to_reference():
    """M0 maps native moving coords into the reference frame, so it applies to SOURCE points."""
    from stare.stage_warp import make_warper

    m0 = [[1, 0, 5.0], [0, 1, -2.0], [0, 0, 1]]
    warp = make_warper({"ref_slide": anhir.REF_NAME,
                        "slides": {anhir.MOV_NAME: {"M0": m0, "mesh": None}}})
    out = warp(anhir.MOV_NAME, np.array([[10.0, 20.0]]), "refined")
    np.testing.assert_allclose(out, [[15.0, 18.0]])


def test_select_and_task_index_order(tmp_path):
    _cover(tmp_path, [(3, "training", "B", "A"), (1, "training", "C", "A"), (2, "evaluation", "D", "A")])
    cases = anhir.select(anhir.load_cases(tmp_path), "training")
    assert [c.case_id for c in cases] == [1, 3]  # sorted by id: array index i is stable
    assert cases[0].tissue == "T" and cases[0].diagonal == 1000.0


def test_score_imputes_missing_and_reports_common(tmp_path, capsys):
    root, out = tmp_path / "data", tmp_path / "out"
    _cover(root, [(0, "training", "S", "R"), (1, "training", "S2", "R")])
    lm = root / "landmarks" / "T_1" / "scale-25pc"
    target = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 100.0]])
    for name, pts in (("R", target), ("S", target + 50.0), ("S2", target + 20.0)):
        anhir.write_landmarks(lm / f"{name}.csv", pts)
    anhir.write_landmarks(out / "stare" / "0.csv", target + 1.0)  # case 1 missing -> imputed

    assert anhir.main(["score", "--data-root", str(root), "--out", str(out)]) == 0
    with open(out / "tables" / "cases.csv", newline="") as fh:
        rows = {(r["method"], r["case_id"]): r for r in csv.DictReader(fh)}
    assert rows[("stare", "0")]["imputed_initial"] == "False"
    assert float(rows[("stare", "0")]["rtre_median"]) == pytest.approx(np.sqrt(2) / 1000)
    assert rows[("stare", "1")]["imputed_initial"] == "True"
    assert float(rows[("stare", "1")]["rtre_median"]) == pytest.approx(
        float(rows[("initial", "1")]["rtre_median"]))
    with open(out / "tables" / "aggregates.csv", newline="") as fh:
        agg = {(r["method"], r["subset"]): r for r in csv.DictReader(fh)}
    assert agg[("stare", "all")]["n_cases"] == "2"
    assert agg[("stare", "common")]["n_cases"] == "1"  # only case 0 was registered


def test_run_records_failure_without_landmarks(tmp_path):
    root = tmp_path / "data"
    _cover(root, [(0, "training", "S", "R")])
    rc = anhir.main(["run", "--data-root", str(root), "--work", str(tmp_path / "w"),
                     "--out", str(tmp_path / "o"), "--task-index", "0", "--workers", "1"])
    assert rc == 1
    rec = json.loads((tmp_path / "o" / "runs" / "0" / "run.json").read_text())
    assert rec["ok"] is False and "landmark" in rec["error"]
