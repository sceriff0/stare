#!/bin/bash
# Run every calibration candidate on the dev_* cases, then lock the winners.
#
#   export REGBENCH_CASES=... REGBENCH_OUT=... [per-method environments, see env.sh]
#   python -m regbench prepare synthetic --suite dev --cases "$REGBENCH_CASES"    # once
#   benchmarks/regbench/slurm/submit_calibrate.sh [--datasets "synthetic semisynth"] \
#       [--methods "valis deeperhistreg"] [-- extra sbatch args]
#
# Results go to $REGBENCH_OUT/calibration, apart from the benchmark's own; the lock is
# $REGBENCH_OUT/calibration/params.lock.json, which `submit.sh --arm recommended` reads.
# Run this BEFORE looking at any test result: that is what makes the choice blind.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

datasets="synthetic semisynth" methods="valis deeperhistreg"
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets) datasets="$2"; shift 2 ;;
        --methods) methods="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
extra=("$@")
main_out="$REGBENCH_OUT"
export REGBENCH_OUT="$main_out/calibration"
mkdir -p "$REGBENCH_OUT/logs"
cd "$REGBENCH_OUT/logs"
alloc=(--cpus-per-task="$REGBENCH_CPUS" --mem="$REGBENCH_MEM" --time="$REGBENCH_TIME")

ids=()
while read -r m label opts; do
    case " $methods " in *" $m "*) ;; *) continue ;; esac
    # shellcheck disable=SC2206
    per_method=($(_regbench_var SBATCH "$m"))
    for ds in $datasets; do
        for cid in $(regbench_py stare list --cases "$REGBENCH_CASES" --dataset "$ds" --ids | grep '^dev_' || true); do
            export ARM_LABEL="$label" ARM_OPTS="$opts"
            j=$(sbatch --parsable --export="ALL,METHOD=$m,DATASET=$ds,CASE_ID=$cid" \
                --job-name="rbcal-$label" "${alloc[@]}" "${extra[@]}" "${per_method[@]}" \
                "$HERE/run_array.sbatch")
            ids+=("${j%%;*}")
        done
    done
    echo "$label: ${opts:-(defaults)}"
done < <(regbench_py stare calibrate --cases "$REGBENCH_CASES" --out "$REGBENCH_OUT" --print-configs)
[ ${#ids[@]} -gt 0 ] || { echo "no dev_* cases prepared under $REGBENCH_CASES"; exit 1; }
dep=$(IFS=:; echo "${ids[*]}")
js=$(sbatch --parsable --export=ALL --dependency=afterany:"$dep" --cpus-per-task="$REGBENCH_CPUS" \
    --mem=32G --time=02:00:00 --job-name=rbcal-lock --output=regbench-calibrate-%j.log "${extra[@]}" \
    --wrap "source '$HERE/env.sh' && regbench_py stare calibrate --cases '$REGBENCH_CASES' --out '$REGBENCH_OUT'")
echo "${#ids[@]} runs submitted; lock job $js -> $REGBENCH_OUT/params.lock.json"
