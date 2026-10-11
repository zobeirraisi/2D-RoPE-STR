#!/usr/bin/env bash
# E3c on the PC (RTX 4090): seeded full-schedule 2D-RoPE (Amendment 3), exactly the Table 1
# configuration (learnable beta, equal split, legacy decoder) with seed 0. Starts after the PC's
# E5 seed-2 pair (third_party/parseq/logs/e5_pc_s2.DONE).
#
# Launch (detached):
#   cd code && setsid nohup bash scripts/e3c_2drope.sh >> logs/e3c_2drope.log 2>&1 < /dev/null &
# Stop:   touch logs/e3c_2drope.STOP; pkill -f "e3c_2drop[e].sh"; pkill -f "run_name full_2drop[e]"
# Power-loss safe: atomic checkpoints every 2,500 iters + resume; a crontab @reboot
# entry re-runs this script (exits if DONE/STOP or already running).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

[ -f logs/e3c_2drope.DONE ] && { echo "E3c already finished"; exit 0; }
[ -f logs/e3c_2drope.STOP ] && { echo "E3c stopped by user (rm logs/e3c_2drope.STOP to allow restarts)"; exit 0; }
exec 9> logs/e3c_2drope.lock
flock -n 9 || { echo "E3c already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }

WAIT_FOR="${WAIT_FOR:-$PWD/../third_party/parseq/logs/e5_pc_s2.DONE}"
i=0
until [ -f "$WAIT_FOR" ]; do
  [ $((i % 30)) -eq 0 ] && stamp "waiting for $WAIT_FOR"
  i=$((i + 1)); sleep 60
done

PY=.venv/bin/python
CK=checkpoints_rev1
RES=results/rev1
RUN=full_2drope_s0
mkdir -p "$CK" "$RES"

RESUME=()
[ -f "$CK/$RUN.pt" ] && RESUME=(--resume "$CK/$RUN.pt")
stamp "E3c $RUN on $(nvidia-smi --query-gpu=name --format=csv,noheader) ${RESUME[*]:-}"
$PY train_2drope_str.py --pe_type 2drope --learnable_freq --dim_split 0.5 --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 \
    --eval_every 0 --save_dir "$CK" --run_name "$RUN" --data_root data_fast "${RESUME[@]}"
$PY evaluate.py --checkpoint "$CK/$RUN.pt" --data_root data_fast --out "$RES/$RUN.json"
stamp "E3c DONE"
touch logs/e3c_2drope.DONE
