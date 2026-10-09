#!/usr/bin/env bash
# R1-3: cross-attention diagnostic (analyze_cross_attention.py) on the E2 factorial models.
# Starts when the local queue has finished; eval only (~2 min per model), runs alongside E4/E5.
#   cd code && setsid nohup bash scripts/e2_xattn.sh >> logs/e2_xattn.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f logs/e2_xattn.DONE ] && { echo "already finished"; exit 0; }
exec 9> logs/e2_xattn.lock
flock -n 9 || { echo "already running"; exit 0; }
stamp() { echo "[$(date '+%F %T')] $*"; }
i=0
until [ -f logs/rev1_queue.DONE ]; do
  [ $((i % 12)) -eq 0 ] && stamp "waiting for logs/rev1_queue.DONE"
  i=$((i + 1)); sleep 300
done
stamp "xattn start"
.venv/bin/python analyze_cross_attention.py --data_root data_fast --out_dir results/rev1/xattn \
    --checkpoints checkpoints_rev1/abl_factorial_*_s[0-9].pt
.venv/bin/python make_tables.py factorial
stamp "xattn DONE"
touch logs/e2_xattn.DONE
