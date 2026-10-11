#!/usr/bin/env bash
# E3b on the PC (RTX 4090): full-schedule 1D sinusoidal baseline, same recipe as E3
# (full_1drope_s0 on gpu3090) and the Table-2 2D-RoPE model. Starts after the PC's
# E5 runs finish (third_party/parseq/logs/e5_pc.DONE).
#
# Launch (detached):
#   cd code && setsid nohup bash scripts/e3b_1dsin.sh >> logs/e3b_1dsin.log 2>&1 < /dev/null &
# Stop:   touch logs/e3b_1dsin.STOP; pkill -f "e3b_1dsi[n].sh"; pkill -f "run_name full_1dsi[n]"
# Power-loss safe: atomic checkpoints every 2,500 iters + resume; a crontab @reboot
# entry re-runs this script (exits if DONE/STOP or already running).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

[ -f logs/e3b_1dsin.DONE ] && { echo "E3b already finished"; exit 0; }
[ -f logs/e3b_1dsin.STOP ] && { echo "E3b stopped by user (rm logs/e3b_1dsin.STOP to allow restarts)"; exit 0; }
exec 9> logs/e3b_1dsin.lock
flock -n 9 || { echo "E3b already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }

WAIT_FOR="${WAIT_FOR:-$PWD/../third_party/parseq/logs/e5_pc.DONE}"
i=0
until [ -f "$WAIT_FOR" ]; do
  [ $((i % 30)) -eq 0 ] && stamp "waiting for $WAIT_FOR"
  i=$((i + 1)); sleep 60
done

PY=.venv/bin/python
CK=checkpoints_rev1
RES=results/rev1
RUN=full_1dsin_s0
mkdir -p "$CK" "$RES"

RESUME=()
[ -f "$CK/$RUN.pt" ] && RESUME=(--resume "$CK/$RUN.pt")
stamp "E3b $RUN on $(nvidia-smi --query-gpu=name --format=csv,noheader) ${RESUME[*]:-}"
$PY train_2drope_str.py --pe_type 1d_sin --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 \
    --eval_every 0 --save_dir "$CK" --run_name "$RUN" --data_root data_fast "${RESUME[@]}"
$PY evaluate.py --checkpoint "$CK/$RUN.pt" --data_root data_fast --out "$RES/$RUN.json"
stamp "E3b DONE"
touch logs/e3b_1dsin.DONE
