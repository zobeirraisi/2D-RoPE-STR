#!/usr/bin/env bash
# E3 on the RTX 3090 server: full-schedule 1D RoPE baseline (same recipe as the
# Table-2 2D-RoPE model). Lives in ~/ROPE_rev1/code on gpu3090.
#
# Launch (detached):
#   cd ~/ROPE_rev1/code && setsid nohup bash scripts/e3_3090.sh >> logs/e3_3090.log 2>&1 < /dev/null &
# Stop:   touch logs/e3_3090.STOP; pkill -f "e3_309[0].sh"; pkill -f "train_2drope_st[r].py"
# Power-loss safe: atomic checkpoints every 2,500 iters + resume; a crontab
# @reboot entry re-runs this script (exits if DONE/STOP or already running).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

[ -f logs/e3_3090.DONE ] && { echo "E3 already finished"; exit 0; }
[ -f logs/e3_3090.STOP ] && { echo "E3 stopped by user (rm logs/e3_3090.STOP to allow restarts)"; exit 0; }
exec 9> logs/e3_3090.lock
flock -n 9 || { echo "E3 already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }

# Wait until the GPU has room (another job may be finishing) — needs ~12 GB.
for i in $(seq 720); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  [ "${free:-0}" -gt 14000 ] && break
  [ $((i % 10)) -eq 1 ] && stamp "waiting for GPU memory (free=${free:-?} MiB)"
  sleep 60
done

PY=.venv/bin/python
CK=checkpoints_rev1
RES=results/rev1
RUN=full_1drope_s0
mkdir -p "$CK" "$RES"

RESUME=()
[ -f "$CK/$RUN.pt" ] && RESUME=(--resume "$CK/$RUN.pt")
stamp "E3 $RUN on $(nvidia-smi --query-gpu=name --format=csv,noheader) ${RESUME[*]:-}"
$PY train_2drope_str.py --pe_type 1drope --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 \
    --eval_every 0 --save_dir "$CK" --run_name "$RUN" --data_root data "${RESUME[@]}"
$PY evaluate.py --checkpoint "$CK/$RUN.pt" --data_root data --out "$RES/$RUN.json"
stamp "E3 DONE"
touch logs/e3_3090.DONE
