#!/bin/bash
#SBATCH --job-name=anhir-prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=anhir-prepare-%j.log
# Rejoin the split image archive and unzip it to $ANHIR_DATA/images (once; ~26 GB extra disk).
set -euo pipefail
source "$(dirname "$0")/env.sh" 2>/dev/null || source "$STARE_REPO/benchmarks/anhir/slurm/env.sh"
mkdir -p "$ANHIR_WORK" "$(dirname "$ANHIR_OUT")"
anhir_py join --data-root "$ANHIR_DATA"
