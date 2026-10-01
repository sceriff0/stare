#!/bin/bash
#SBATCH --job-name=anhir-stare
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=anhir-stare-%A_%a.log
# One ANHIR training case per array task: --task-index is SLURM_ARRAY_TASK_ID into the
# case list sorted by id (`anhir.py list` prints its length). Submit through submit.sh.
set -uo pipefail
source "$STARE_REPO/benchmarks/anhir/slurm/env.sh"
mkdir -p "$ANHIR_WORK" "$ANHIR_OUT"
anhir_py run --data-root "$ANHIR_DATA" --work "$ANHIR_WORK" --out "$ANHIR_OUT" \
    --status training --task-index "$SLURM_ARRAY_TASK_ID" \
    --proxy "$PROXY" --workers "$SLURM_CPUS_PER_TASK" ${ANHIR_RUN_ARGS:-}
# A failed registration is recorded in runs/<id>/run.json and scored at the initial pose,
# so the task exits 0 either way; grep the logs for FAILED.
exit 0
