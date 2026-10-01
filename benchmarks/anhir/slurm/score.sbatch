#!/bin/bash
#SBATCH --job-name=anhir-score
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=anhir-score-%j.log
# Score initial / bUnwarpJ / stare_rigid / stare against the training target landmarks.
set -euo pipefail
source "$STARE_REPO/benchmarks/anhir/slurm/env.sh"
anhir_py score --data-root "$ANHIR_DATA" --out "$ANHIR_OUT"
grep -l '"ok": false' "$ANHIR_OUT"/runs/*/run.json 2>/dev/null | sed 's/^/FAILED: /' || true
