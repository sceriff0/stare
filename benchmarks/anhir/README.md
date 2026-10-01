# ANHIR benchmark for STARE

Scores STARE on the public ANHIR challenge (Borovec et al. 2020, IEEE TMI,
doi:10.1109/TMI.2020.2986331) with the challenge's own metrics
(<https://anhir.grand-challenge.org/Performance_Metrics/>). It compares four methods:

- **initial**: no registration, the identity transform.
- **bunwarpj**: the challenge's published bUnwarpJ baseline.
- **stare_rigid**: STARE's coarse stage only, the global anchor `M0`.
- **stare**: the full STARE transform, `M0` plus the mesh.

This lives on the `benchmarking` branch. It is a harness, not part of the package.

## What is measured

ANHIR scores landmarks, not pixels. STARE registers the case's **source** image (moving) onto
its **target** image (reference). Its manifest is a forward map from moving to reference, so
the source landmarks are pushed through `stare.stage_warp.make_warper(manifest)` and compared
with the target landmarks.

| number | definition |
|---|---|
| rTRE | ‖warped source − target‖ / image diagonal, using the cover table's diagonal |
| per case | median, mean and max rTRE; the **median** is the headline, as in the challenge |
| robustness | fraction of landmarks brought closer than the unregistered pose |
| rank | each case's methods ranked by median rTRE (1 = best) |

A failed registration still counts: it is scored at the initial pose (`imputed_initial`), as
the challenge does. The scorer also reports a **`common`** subset: the cases every method
actually registered.

**About the bUnwarpJ baseline.** Its published output covers 84 of the 230 training cases,
and all 84 were made on an **older landmark release**: the `source_landmarks.csv` in each
`BmUnwarpJ/<id>/` folder has a different count and different points from the current archive.
Landmarks are paired by index, so those outputs cannot be scored against the current targets.
The scorer checks every baseline case and drops the method, with a note, when no case is
usable, which is the situation today. Each case's `problem` column records why a method had
no usable output.

Only the 230 **training** cases have public target landmarks. The 251 evaluation cases are
scored by the challenge server, and none of their numbers come from this harness.

## Data layout (`--data-root`)

```
dataset_medium.csv                     cover table (481 cases)
dataset_medium.z01..z05, .zip          split image archive (12.8 GB)
images/<set>/scale-25pc/*.jpg          after `anhir.py join`
landmarks/<set>/scale-*/*.csv          the separate landmark download (needs a challenge login)
BmUnwarpJ/<case_id>/                   the bUnwarpJ baseline
```

The landmark download is itself named `dataset_medium.zip`, the same name as the image
archive's last part. Rename it (here: `landmarks.zip`) before putting it next to the images,
then unzip it into `landmarks/`. `run` and `score` both need the archive's landmarks.

## Brightfield to STARE input

STARE reads one nuclear channel. Each JPEG is converted once, in a cache shared safely by
array tasks, to a one-channel tiled OME-TIFF:

- `--proxy lum` (default): inverted luminance. Dark nuclei become bright and the white
  background goes to 0, like DAPI.
- `--proxy hema`: the haematoxylin channel from `skimage.color.rgb2hed`. Its matrix is
  H&E-DAB, so it only approximates the kidney stains (PAS and others).

ANHIR pairs mix stains (H&E ↔ IHC). STARE was built for a channel that is identical in every
round, so ANHIR is out of domain for it, and its numbers here are a stress test.

## Run locally

```bash
git switch benchmarking
export PYTHONPATH=src
D=challenge/anhir                              # gitignored; the data lives here
python benchmarks/anhir/anhir.py join --data-root $D                     # once
python benchmarks/anhir/anhir.py run  --data-root $D --work /tmp/anhir_work \
    --out results/anhir --case-id 197 --workers 8                         # one case
python benchmarks/anhir/anhir.py score --data-root $D --out results/anhir
PYTHONPATH=src pytest benchmarks/anhir                                    # harness tests
```

Each run writes these files under `--out`:

- `stare/<id>.csv` and `stare_rigid/<id>.csv`: warped source landmarks, in ANHIR format.
- `runs/<id>/run.json`: timings, the solve report and any error.
- `runs/<id>/manifest.json`: the transform.

`--keep-work` also keeps the M0, the tile plan and the control JSONs. `--stitch` also writes
the registered image, for visual QC.

## Run on SLURM

```bash
export STARE_REPO=$HOME/stare               # this repo, on the benchmarking branch
export ANHIR_DATA=/path/to/anhir            # the layout above
export ANHIR_WORK=$SCRATCH/anhir_work
export STARE_SIF=$HOME/stare.sif            # optional: apptainer pull stare.sif docker://bolt3x/mirage-stare:1.0.0
benchmarks/anhir/slurm/submit.sh --prepare -- --partition=<cpu-partition>
```

`submit.sh` chains three jobs:

1. The one-off archive join (only with `--prepare`).
2. An array of 230 tasks, one training case each, with 8 CPUs, 32 GB and up to 1 h per task.
3. The scorer, which runs once the array finishes.

Results land in `$ANHIR_OUT`, which defaults to `~/anhir_results/stare-<commit>`, so runs of
different commits never overwrite each other. `ANHIR_RUN_ARGS` passes extra `run` flags, for
example `ANHIR_RUN_ARGS="--stride 64"`. `PROXY=hema` switches the nuclear proxy.

## Outputs of `score` (`$ANHIR_OUT/tables/`)

- **`cases.csv`**: one row per (case, method). Columns are median, mean and max rTRE; median
  TRE in px; robustness; rank; time; and `imputed_initial`.
- **`aggregates.csv`**: one row per (method, subset), where subset is `all`, `common` or
  `tissue:<name>`. Columns are the average and the median of the per-case median rTRE,
  average max rTRE, average robustness, average rank and median time.
