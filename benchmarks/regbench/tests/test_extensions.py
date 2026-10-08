"""Deformation families, the real-image suite, HyReCo, the scorer's references and
statistics, calibration, and the ANHIR submission layout."""

import argparse
import csv
import json

import numpy as np
import pytest
from regbench import calibrate, score
from regbench.cases import (
    Case,
    list_cases,
    load_case,
    load_points,
    result_dir,
    write_case,
    write_json,
    write_npz,
)
from regbench.datasets import hyreco, semisynth, synthetic
from regbench.imageio import read_rgb_reduced, write_nuclear
from regbench.methods import deeperhistreg, run_case

SMALL = dict(n=768, theta=1.0, shift=(6.0, -4.0), seed=5)
FIELDS = {
    "multiscale": dict(field="multiscale", scales=(600.0, 200.0, 80.0), rms=(4.0, 2.0, 0.5)),
    "grid": dict(field="grid", grid_sd=4.0),
    "bumps": dict(field="bumps", bump_sigma=40.0, bump_count=6),
    "seams": dict(field="seams", seam_tile=256, seam_sd=3.0),
}


def _spec(**kw):
    return {**synthetic._BASE, "family": "t", **SMALL, **kw}


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_field_moves_the_image_as_the_map_says(field):
    """A nucleus at ``g(p)`` in the reference is at ``p`` in the moving slide."""
    spec = _spec(noise=0.0, **FIELDS[field])
    ref, mov, scene = synthetic.make_pair(spec)
    g = synthetic.forward_map(spec)
    rng = np.random.default_rng(0)
    p = rng.uniform(64, 700, (4000, 2))
    if field == "seams":  # away from a jump, where the bicubic sample mixes two tiles
        p = p[(np.abs((p + 8) % 256 - 8) > 8).all(axis=1)]
    q = g(p)
    ok = ((q > 8) & (q < 760)).all(axis=1)
    p, q = p[ok], q[ok]
    from scipy.ndimage import map_coordinates

    a = map_coordinates(ref.astype(float), [q[:, 1], q[:, 0]], order=3)
    b = map_coordinates(mov.astype(float), [p[:, 1], p[:, 0]], order=3)
    assert np.corrcoef(a, b)[0, 1] > 0.995
    assert np.abs(g(p) - synthetic.forward_map(_spec())(p)).max() > 1.0  # the field is not nil


@pytest.mark.parametrize("field", ["multiscale", "grid", "bumps"])
def test_smooth_fields_do_not_fold(field):
    spec = synthetic.SUITES["full"][f"{field}_2" if field != "bumps" else "bumps_0"]
    xs = np.arange(0, spec["n"], 8.0)
    q = synthetic.forward_map(spec)(np.stack(np.meshgrid(xs, xs), axis=-1))
    jac = (np.gradient(q[..., 0], 8, axis=1) * np.gradient(q[..., 1], 8, axis=0)
           - np.gradient(q[..., 0], 8, axis=0) * np.gradient(q[..., 1], 8, axis=1))
    assert jac.min() > 0.3


def test_dev_seeds_are_disjoint_from_every_scored_suite():
    dev = {s["seed"] for s in synthetic.SUITES["dev"].values()}
    rest = {s["seed"] for k in ("full", "scale", "ci", "test") for s in synthetic.SUITES[k].values()}
    assert not dev & rest
    assert all(name.startswith("dev_") for name in synthetic.SUITES["dev"])


# ── the scorer's references ───────────────────────────────────────────────────
def test_truth_is_the_ceiling_and_null_is_chance(cases_root, synthetic_cases, tmp_path):
    out = tmp_path / "out"
    done = score.score(cases_root, out, ["synthetic"])
    by = {(r["case_id"], r["method"]): r for r in done["cells"]}
    for name in synthetic_cases:
        truth, init = by[name, "truth"], by[name, "initial"]
        assert truth["dice_matched"] > 0.85 and truth["pair_fraction"] > 0.9
        assert truth["dice_matched"] > init["dice_matched"]
        assert truth["dice_null"] < truth["dice_matched"] - 0.2
    agg = [r for r in done["cells_agg"] if r["method"] == "truth" and r["subset"] == "all"][0]
    assert agg["avg_dice_matched_lo"] <= agg["avg_dice_matched"] <= agg["avg_dice_matched_hi"]
    assert not any(r["method"] == "truth" for r in done["landmarks"])
    assert "self-consistency" in (out / "tables" / "summary.md").read_text()


def test_bootstrap_interval_and_pairwise_test():
    lo, hi = score.bootstrap_ci(np.arange(50.0), np.median)
    assert lo < 24.5 < hi and hi - lo < 25
    assert all(np.isnan(score.bootstrap_ci([1.0, 2.0], np.mean)))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        base = rng.uniform(1, 2)
        for m, d in (("a", 0.0), ("b", 0.5), ("c", 0.5 + rng.normal(0, 0.01)), ("truth", -1.0)):
            rows.append({"dataset": "d", "case_id": str(i), "method": m, "v": base + d})
    out = {(r["method_a"], r["method_b"]): r for r in score.pairwise(rows, "v")}
    assert set(out) == {("a", "b"), ("a", "c"), ("b", "c")}  # the pseudo-method is not compared
    assert out["a", "b"]["a_better"] == 20 and out["a", "b"]["p_holm"] < 1e-3
    assert out["b", "c"]["p_holm"] > 0.05
    assert out["a", "b"]["p_holm"] >= out["a", "b"]["p_wilcoxon"]
    flipped = score.pairwise(rows, "v", higher_is_better=True)
    assert flipped[0]["a_better"] == 0 and flipped[0]["b_better"] == 20


def test_dev_cases_are_scored_only_on_request(tmp_path):
    root = tmp_path / "cases"
    spec = {**synthetic._BASE, "family": "dev", "n": 512, "seed": 3, "theta": 1.0}
    synthetic.prepare_one(root, "dev_a", spec)
    synthetic.prepare_one(root, "a", {**spec, "family": "x"})
    ids = lambda dev: {r["case_id"] for r in score.score(root, tmp_path / "o", dev=dev)["landmarks"]}
    assert ids(False) == {"a"} and ids(True) == {"dev_a"}
    assert (tmp_path / "o" / "tables_dev" / "summary.md").exists()


# ── real images under a known map ─────────────────────────────────────────────
def _semisynth_args(image, cases, **kw):
    base = dict(image=str(image), channel=0, mov_image=None, mov_channel=0, name="slide", n=512,
                windows=2, diameter=synthetic.NUCLEUS_DIAMETER, noise=1.0, pixel_size_um=0.5,
                residual_um=2.0,
                cases=str(cases), index=None, workers=1, force=False)
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture(scope="module")
def real_like(tmp_path_factory):
    """A slide with tissue in part of it only, standing in for a real nuclear channel."""
    ref, _, _ = synthetic.make_pair({**synthetic._BASE, "family": "t", "n": 1536, "seed": 8,
                                     "keep": 0.5})
    path = tmp_path_factory.mktemp("real") / "slide.ome.tif"
    write_nuclear(path, ref, 0.5)
    return path


def test_semisynth_cases_carry_the_truth(real_like, tmp_path):
    a = _semisynth_args(real_like, tmp_path / "cases")
    jobs, noise = semisynth.plan(a)
    assert noise > 0
    names = [j[0] for j in jobs]
    assert sum(n.startswith("dev_") for n in names) == len(semisynth.FAMILIES)
    assert len(names) == 3 * len(semisynth.FAMILIES)
    dev_origin = {j[2] for j in jobs if j[0].startswith("dev_")}
    assert not dev_origin & {j[2] for j in jobs if not j[0].startswith("dev_")}
    assert len({j[1]["seed"] for j in jobs}) == len(jobs)

    name, spec, origin = next(j for j in jobs if j[0].endswith("w0_multiscale"))
    semisynth.prepare_one(a.cases, name, spec, a, origin, noise, 0.5)
    case = load_case(tmp_path / "cases" / "semisynth" / name)
    assert case.pixel_size_um == 0.5 and case.extra["variant"] == "same"
    pts = load_points(case)
    assert len(pts["landmarks_mov"]) > 50
    np.testing.assert_allclose(synthetic.forward_map(spec)(pts["landmarks_mov"]),
                               pts["landmarks_ref"])
    done = score.score(a.cases, tmp_path / "out", ["semisynth"])
    by = {r["method"]: r for r in done["cells"]}
    assert by["truth"]["dice_matched"] > 0.8 > by["initial"]["dice_matched"]
    assert by["truth"]["displacement_px_p50"] < 1.0
    # a real-image case under a literature-derived map is headline, and aggregated as such
    assert {r["tier"] for r in done["cells"]} == {"headline"}
    assert any(r["subset"] == "headline" for r in done["cells_agg"])
    assert "## Headline" in (tmp_path / "out" / "tables" / "summary.md").read_text()


def test_multiscale_amplitude_is_the_measured_residual():
    """RMS displacement of the headline multiscale field is --residual-um, in pixels."""
    n, px = 4096, 0.325
    kw = semisynth.family_spec("multiscale", n, px, 2.0, seed=1)
    assert kw["scales"] == (512.0, 256.0, 128.0)
    spec = {**synthetic._BASE, "n": n, "seed": 1, **kw}
    xs = np.linspace(0, n - 1, 200)
    dx, dy = synthetic.extra_field(spec)(*np.meshgrid(xs, xs))
    assert abs(np.sqrt(np.mean(dx**2 + dy**2)) * px - 2.0) < 0.3
    sd = semisynth.family_spec("grid", n, px, 2.0, seed=1)["grid_sd"]
    assert 3.0 <= sd <= 7.0


def test_tiers_and_folding():
    def case(dataset, group):
        return Case(dataset=dataset, case_id="c", group=group, ref_nuclear="", mov_nuclear="",
                    ref_image="", mov_image="", modality="fluorescence", diagonal=1.0)

    assert score.tier_of(case("hyreco", "HE-PHH3")) == "headline"
    assert score.tier_of(case("multiplex", "P1")) == "headline"
    assert score.tier_of(case("semisynth", "grid")) == "headline"
    assert score.tier_of(case("semisynth", "seams")) == "secondary"
    assert score.tier_of(case("synthetic", "grid")) == "secondary"
    assert score.tier_of(case("anhir", "COAD")) == "secondary"

    from regbench.methods import jacobian_grid

    xy, shape, step = jacobian_grid((300, 400))
    pack = lambda w: {"grid": w, "grid_shape": np.array(shape), "grid_step": np.array(step)}  # noqa: E731
    smooth = score.jacobian_stats(pack(xy * 1.1 + 3.0))
    assert smooth["fold_pct"] == 0.0 and smooth["sd_log_jac"] < 1e-6
    folded = xy.copy()
    left = xy[:, 0] < 200
    folded[left, 0] = 200 - xy[left, 0]  # the left half is mirrored: it folds
    out = score.jacobian_stats(pack(folded))
    assert 40 < out["fold_pct"] < 60
    assert np.isnan(score.jacobian_stats(None)["fold_pct"])


def test_semisynth_reference_is_the_untouched_window(real_like, tmp_path):
    import tifffile

    a = _semisynth_args(real_like, tmp_path / "cases")
    name, spec, origin = semisynth.plan(a)[0][0]
    semisynth.prepare_one(a.cases, name, spec, a, origin, 0.0, 0.5)
    x, y = origin
    whole = np.squeeze(tifffile.imread(real_like))
    got = np.squeeze(tifffile.imread(tmp_path / "cases" / "semisynth" / name / "ref.ome.tif"))
    np.testing.assert_array_equal(got, whole[y : y + 512, x : x + 512])


# ── HyReCo ────────────────────────────────────────────────────────────────────
def _wsi(path, rgb, um_per_px):
    import tifffile

    per_cm = 1e4 / um_per_px
    tifffile.imwrite(path, rgb, photometric="rgb", tile=(256, 256), bigtiff=True,
                     resolution=(per_cm, per_cm), resolutionunit="CENTIMETER")


def test_hyreco_landmarks_are_converted_from_millimetres(tmp_path):
    root, um = tmp_path / "data", 0.25
    rng = np.random.default_rng(0)
    pts = {"HE": rng.uniform(20, 480, (12, 2)), "PHH3": rng.uniform(20, 480, (12, 2))}
    for stain, xy in pts.items():
        (root / stain).mkdir(parents=True)
        _wsi(root / stain / "29.tif", rng.integers(120, 255, (512, 640, 3), dtype=np.uint8), um)
        rows = ["x,y,z"] + [f"{x * um / 1000:.9f},{y * um / 1000:.9f},0.0" for x, y in xy]
        (root / stain / "29.csv").write_text("\n".join(rows))
    a = argparse.Namespace(data_root=str(root), ref_dir=None, mov_dir=None, pixel_size_um=None,
                           cases=str(tmp_path / "cases"), index=None, workers=1, force=False)
    assert hyreco.prepare(a) == 0
    (case,) = list_cases(a.cases, "hyreco")
    assert case.case_id == "29" and case.modality == "brightfield"
    assert case.ref_hw == [512, 640] and abs(case.pixel_size_um - um) < 1e-6
    got = load_points(case)
    np.testing.assert_allclose(got["landmarks_mov"], pts["PHH3"], atol=1e-3)
    np.testing.assert_allclose(got["landmarks_ref"], pts["HE"], atol=1e-3)
    import tifffile

    assert np.squeeze(tifffile.imread(case.ref_nuclear)).shape == (512, 640)
    rgb, f = read_rgb_reduced(case.ref_image, 200)
    assert f == 4 and rgb.shape == (128, 160, 3)


def test_hyreco_refuses_a_wrong_pixel_size(tmp_path):
    root = tmp_path / "data"
    for stain in ("HE", "PHH3"):
        (root / stain).mkdir(parents=True)
        _wsi(root / stain / "8.tif", np.full((256, 256, 3), 200, np.uint8), 0.25)
        (root / stain / "8.csv").write_text("0.05,0.05,0\n")
    with pytest.raises(RuntimeError, match="wrong pixel size"):
        hyreco.prepare_one(tmp_path / "cases", "8", root / "HE", root / "PHH3", pixel_size_um=0.05)


# ── method arms ───────────────────────────────────────────────────────────────
def test_inverse_warper_inverts_a_rotated_smooth_map():
    th = np.radians(25.0)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def forward(xy):
        return xy @ rot.T + [40.0, -15.0] + 6.0 * np.sin(xy[:, ::-1] / 90.0)

    inv = deeperhistreg.inverse_warper(forward, (1000, 1200))
    p = np.random.default_rng(1).uniform(0, 1000, (500, 2))
    np.testing.assert_allclose(inv(forward(p)), p, atol=0.05)


def test_calibration_budget_is_equal_and_the_lock_picks_the_best():
    assert {len(v) for v in calibrate.CANDIDATES.values()} == {calibrate.BUDGET}
    assert "stare" not in calibrate.CANDIDATES
    rows = []
    for m, lab, _ in calibrate.configs():
        for variant, base in ((lab, 0.01), (f"{lab}_rigid", 0.05)):
            for c in range(3):
                good = lab.endswith("_c3") and variant == lab
                rows.append({"method": variant, "case_id": f"dev_{c}", "imputed_initial": False,
                             "rtre_median": 0.001 if good else base, "tre_px_median": 1.0})
    lock = calibrate.choose(rows)
    for m in calibrate.CANDIDATES:
        assert lock[m]["label"] == f"{m}_c3" and lock[m]["opts"] == calibrate.CANDIDATES[m][3]
        assert len(lock[m]["candidates"]) == 2 * calibrate.BUDGET


def test_recommended_arm_falls_back_to_the_documented_setting(tmp_path):
    assert calibrate.recommended("stare")[0] == {}
    assert calibrate.recommended("valis", tmp_path / "none.json")[0] == {"micro_fraction": "0.25"}
    lock = tmp_path / "params.lock.json"
    lock.write_text(json.dumps({"valis": {"label": "valis_c2", "opts": {"micro_fraction": "1.0"}}}))
    opts, source = calibrate.recommended("valis", lock)
    assert opts == {"micro_fraction": "1.0"} and "valis_c2" in source


# ── ANHIR submission ──────────────────────────────────────────────────────────
def test_anhir_submission_layout(tmp_path):
    from regbench.datasets import anhir

    data, cases, out = tmp_path / "data", tmp_path / "cases", tmp_path / "out"
    data.mkdir()
    head = ["", "Source image", "Target image", "Source landmarks", "Target landmarks", "status",
            "Image diagonal [pixels]"]
    with open(data / "dataset_medium.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for cid in (3, 4, 5):
            w.writerow([cid, "a.jpg", "b.jpg", "a.csv", "b.csv", "evaluation", 1000.0])
    src = np.array([[10.0, 20.0], [30.0, 40.0]])
    for cid in (3, 4):  # case 5 was never prepared
        case = Case(dataset="anhir", case_id=str(cid), group="t", ref_nuclear="", mov_nuclear="",
                    ref_image="", mov_image="", modality="brightfield", diagonal=1000.0)
        write_case(cases, case, src)  # hidden targets: source landmarks only
        d = result_dir(out, "stare", case)
        ok = cid == 3
        write_json(d / "run.json", {"ok": ok, "register_s": 120.0})
        if ok:
            write_npz(d / "warped.npz", landmarks=src + [1.0, np.nan], cells_xy=np.empty((0, 2)))
    dest = anhir.export_submission(cases, out, data, "stare", tmp_path / "sub")
    with open(dest / "registration-results.csv", newline="") as fh:
        rows = {int(r[""]): r for r in csv.DictReader(fh)}
    assert rows[3]["Execution time [minutes]"] == "2.0000"
    assert rows[4]["Execution time [minutes]"] == "" and rows[5]["Warped source landmarks"] == ""
    h = anhir.harness()
    # a non-finite warped point and an unregistered case both stay at the source position
    np.testing.assert_allclose(h.read_landmarks(dest / rows[3]["Warped source landmarks"]), src)
    np.testing.assert_allclose(h.read_landmarks(dest / rows[4]["Warped source landmarks"]), src)
    # and the scorer does not invent landmark rows for hidden targets
    assert score.score(cases, out, ["anhir"])["landmarks"] == []


def test_stare_on_a_local_bump_case(tmp_path):
    """A full run on one of the new families, scored against truth and the two references."""
    root = tmp_path / "cases"
    spec = _spec(n=1024, theta=1.0, shift=(8.0, -5.0), field="bumps", bump_sigma=120.0,
                 bump_count=4, seed=11)
    synthetic.prepare_one(root, "bumps_t", spec)
    case = load_case(root / "synthetic" / "bumps_t")
    assert run_case("stare", case, tmp_path / "out", workers=1)
    done = score.score(root, tmp_path / "out", ["synthetic"])
    lm = {r["method"]: r for r in done["landmarks"]}
    ce = {r["method"]: r for r in done["cells"]}
    assert lm["stare_rigid"]["tre_px_median"] < lm["initial"]["tre_px_median"]
    assert ce["truth"]["dice_matched"] >= ce["stare"]["dice_matched"] - 0.02
    assert ce["stare"]["dice_null"] < ce["stare"]["dice_matched"]
