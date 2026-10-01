#!/bin/bash
# Submit the whole ANHIR benchmark: [prepare] -> array of cases -> score.
#
#   export STARE_REPO=... ANHIR_DATA=... [ANHIR_WORK=... ANHIR_OUT=... STARE_SIF=... PROXY=...]
#   benchmarks/anhir/slurm/submit.sh [--prepare] [--max-parallel 40] [-- extra sbatch args]
#
# --prepare     also run the one-off archive join/unzip first (skip once images/ exists)
# extra sbatch args (after --) go to every job, e.g. `-- --partition=cpu --account=lab`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export STARE_REPO="${STARE_REPO:-$(cd "$HERE/../../.." && pwd)}"
source "$HERE/env.sh"

prepare=0 maxpar=40
while [ $# -gt 0 ]; do
    case "$1" in
        --prepare) prepare=1; shift ;;
        --max-parallel) maxpar="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
mkdir -p "$ANHIR_WORK" "$ANHIR_OUT" "$ANHIR_OUT/logs"
cd "$ANHIR_OUT/logs"

dep=()
if [ "$prepare" = 1 ]; then
    jp=$(sbatch --parsable --export=ALL "${extra[@]}" "$HERE/prepare.sbatch")
    echo "prepare: $jp"
    dep=(--dependency=afterok:"$jp")
fi

n=$(anhir_py list --data-root "$ANHIR_DATA" --status training)
ja=$(sbatch --parsable --export=ALL --array="0-$((n - 1))%$maxpar" "${dep[@]}" "${extra[@]}" \
    "$HERE/run_array.sbatch")
echo "cases:   $ja  ($n training cases, at most $maxpar at once)"
js=$(sbatch --parsable --export=ALL --dependency=afterany:"${ja%%;*}" "${extra[@]}" \
    "$HERE/score.sbatch")
echo "score:   $js"
echo "results -> $ANHIR_OUT  (logs in $ANHIR_OUT/logs)"
