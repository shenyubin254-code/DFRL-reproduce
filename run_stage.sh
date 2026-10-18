#!/usr/bin/env bash
# Convenience entry point for stage 1; never starts on installation.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != '--resume' ) ]]; then
    echo 'Usage: bash run_stage.sh [--resume]'; exit 2
fi
if [[ -z "${DFRL_PYTHON:-}" && -x /home/shen/miniconda3/envs/dfrl39/bin/python ]]; then
    export DFRL_PYTHON=/home/shen/miniconda3/envs/dfrl39/bin/python
fi
exec bash run_group.sh qed 0.6 "$@"
