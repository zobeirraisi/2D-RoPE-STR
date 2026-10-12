#!/usr/bin/env bash
# E4 extension (IJDAR revision, Amendment 4): geometric robustness of the full-schedule models
# (E3 1D RoPE, E3b 1D sinusoidal, E3c seeded 2D-RoPE; June released model as a descriptive extra).
# Evaluation only, same protocol as E4. Starts when E3b and E3c have finished, writes
# results/rev1/geom/<run>.json. Re-runnable: finished checkpoints are skipped.
# full_1drope_s0.pt was copied from gpu3090 on 2026-10-12 (md5 9d620ec75042705038532ee7cb775655).
#
# Launch (detached):
#   cd code && setsid nohup bash scripts/e4_full.sh >> logs/e4_full.log 2>&1 < /dev/null &
# Stop:   touch logs/e4_full.STOP; pkill -f "e4_ful[l].sh"; pkill -f "eval_geometri[c].py"
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
[ -f logs/e4_full.DONE ] && { echo "E4-full already finished"; exit 0; }
[ -f logs/e4_full.STOP ] && { echo "E4-full stopped by user (rm logs/e4_full.STOP to allow restarts)"; exit 0; }
exec 9> logs/e4_full.lock
flock -n 9 || { echo "E4-full already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }
i=0
until [ -f logs/e3b_1dsin.DONE ] && [ -f logs/e3c_2drope.DONE ]; do
  [ $((i % 12)) -eq 0 ] && stamp "waiting for logs/e3b_1dsin.DONE and logs/e3c_2drope.DONE"
  i=$((i + 1)); sleep 300
done

stamp "E4-full start"
.venv/bin/python eval_geometric.py --data_root data_fast --out_dir results/rev1/geom \
    --checkpoints checkpoints_rev1/full_1drope_s0.pt checkpoints_rev1/full_1dsin_s0.pt \
                  checkpoints_rev1/full_2drope_s0.pt checkpoints/full_2drope_best.pt
stamp "E4-full DONE"
touch logs/e4_full.DONE
