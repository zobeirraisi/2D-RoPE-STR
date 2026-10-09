#!/usr/bin/env bash
# E4 (IJDAR revision, R1-4): geometric robustness of the E1 (PE ablation) and E2 (factorial)
# models. Starts when the local queue has finished (logs/rev1_queue.DONE), evaluates every
# final (non-_best) checkpoint, and writes results/rev1/geom/<run>.json. Re-runnable: finished
# checkpoints are skipped.
#
# Launch (detached):
#   cd code && setsid nohup bash scripts/e4_geom.sh >> logs/e4_geom.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
[ -f logs/e4_geom.DONE ] && { echo "E4 already finished"; exit 0; }
[ -f logs/e4_geom.STOP ] && { echo "E4 stopped by user"; exit 0; }
exec 9> logs/e4_geom.lock
flock -n 9 || { echo "E4 already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }
i=0
until [ -f logs/rev1_queue.DONE ]; do
  [ $((i % 12)) -eq 0 ] && stamp "waiting for logs/rev1_queue.DONE"
  i=$((i + 1)); sleep 300
done

stamp "E4 start"
.venv/bin/python eval_geometric.py --data_root data_fast --out_dir results/rev1/geom \
    --checkpoints checkpoints_rev1/abl_pe_*_s[0-9].pt checkpoints_rev1/abl_factorial_*_s[0-9].pt
stamp "E4 DONE"
touch logs/e4_geom.DONE
