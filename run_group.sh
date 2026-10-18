#!/usr/bin/env bash
# One full-data, seed-0 setting at a time. Never automatically starts the next group.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run_group.sh {qed|plogp|drd2|'qed&drd2'} {0.4|0.6} [--resume]"; exit 2
fi
objective="$1"; similarity="$2"; shift 2
case "$objective" in qed|plogp|drd2|'qed&drd2') ;; *) exit 2 ;; esac
case "$similarity" in 0.4|0.6) ;; *) exit 2 ;; esac
if [[ $# -eq 1 && "$1" != '--resume' ]]; then exit 2; fi
cd "$(dirname "$0")"
mkdir -p logs
# Serialize all groups on the single GPU, including run_stage.sh.
exec 9>logs/benchmark_gpu.lock
flock -n 9 || { echo 'Another group is already running through this launcher.'; exit 1; }
tag="${objective//&/_}_${similarity/./}_seed0"
python_bin="${DFRL_PYTHON:-python}"
log="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"
{
    date -Is
    printf 'WSL boot ID: '; cat /proc/sys/kernel/random/boot_id
    "$python_bin" -u run_benchmark.py --objectives "$objective" --similarities "$similarity" \
        --runs 1 --seed_start 0 --epochs_test 20 --swarm_num 20 \
        --spa_max_length 50 --output_dir "benchmark_results/$tag" "$@"
} 2>&1 | tee -a "$log"
