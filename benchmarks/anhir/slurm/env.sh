# shellcheck shell=bash
# Sourced by every ANHIR sbatch script. Edit the paths, or export them before `submit.sh`.
#
#   STARE_REPO   this repository, checked out on the `benchmarking` branch
#   ANHIR_DATA   folder holding dataset_medium.csv, the split archive, BmUnwarpJ/, landmarks/
#   ANHIR_WORK   scratch for the converted OME-TIFFs (~1 GB per distinct image per proxy)
#   ANHIR_OUT    results: warped landmarks, per-case run.json, tables/
#   STARE_SIF    optional: an Apptainer/Singularity image with STARE's dependencies, e.g.
#                built once with `apptainer pull stare.sif docker://bolt3x/mirage-stare:1.0.0`.
#                Leave empty to use the current environment (see CONDA_ENV).
#   CONDA_ENV    optional: a conda env to activate when STARE_SIF is empty
#   PROXY        lum (inverted luminance, default) or hema (haematoxylin deconvolution)

: "${STARE_REPO:=$HOME/stare}"
: "${ANHIR_DATA:=$HOME/anhir}"
: "${ANHIR_WORK:=${SCRATCH:-$HOME/scratch}/anhir_work}"
: "${ANHIR_OUT:=$HOME/anhir_results/stare-$(git -C "$STARE_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}"
: "${STARE_SIF:=}"
: "${CONDA_ENV:=}"
: "${PROXY:=lum}"
export STARE_REPO ANHIR_DATA ANHIR_WORK ANHIR_OUT STARE_SIF CONDA_ENV PROXY

# The repository's source wins over any STARE installed in the image or env, so the
# benchmark always measures the checked-out code.
export PYTHONPATH="$STARE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # parallelism is the tile pool

anhir_py() {
    if [ -n "$STARE_SIF" ]; then
        local bind
        bind="$STARE_REPO,$ANHIR_DATA,$ANHIR_WORK,$(dirname "$ANHIR_OUT")"
        "$(command -v apptainer || command -v singularity)" exec --cleanenv \
            --env "PYTHONPATH=$PYTHONPATH,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1" \
            --bind "$bind" "$STARE_SIF" python "$STARE_REPO/benchmarks/anhir/anhir.py" "$@"
    else
        if [ -n "$CONDA_ENV" ]; then
            # shellcheck disable=SC1091
            source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
        fi
        python "$STARE_REPO/benchmarks/anhir/anhir.py" "$@"
    fi
}
