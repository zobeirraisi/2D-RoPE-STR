#!/usr/bin/env bash
# E5 (IJDAR revision, R2-3 / R1-2): PARSeq with its learned additive position embedding
# vs. 2D-RoPE in the ViT encoder, everything else unchanged. Trained on the same MJ+ST
# LMDBs as our model, official synthetic recipe (20 epochs x 0.20496 of synth, batch 384,
# 2 GPUs) reproduced on ONE GPU: 2-step gradient accumulation (effective batch 768) and
# model.lr rescaled so the optimiser lr equals the 2-GPU value (7e-4*sqrt(2)*384/256);
# swa_base_lr keeps the official SWA lr. Evaluated from final.ckpt (end of training).
#
# Usage (detached):
#   setsid nohup bash scripts/e5_parseq.sh <tag> rope2d:0 learned:1 ... >> <parseq>/logs/e5_<tag>.log 2>&1 < /dev/null &
# Env: WAIT_FOR=<file>  start only after this file exists (e.g. the E3 or queue DONE marker)
#      NUM_WORKERS=12   PARSEQ_DIR=<path> (default ../third_party/parseq next to code/)
# Stop: touch <parseq>/logs/e5_<tag>.STOP; pkill -f "e5_parse[q].sh"; pkill -f "parseq.*train.p[y]"
# Power-loss safe: Lightning writes last.ckpt at every validation (1000 batches); a rerun
# resumes from it, skips finished runs (runs/<run>/DONE), and exits if DONE/STOP/running.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
PQ="${PARSEQ_DIR:-$CODE/../third_party/parseq}"
cd "$PQ"
mkdir -p logs runs results

TAG=$1; shift
[ -f "logs/e5_$TAG.DONE" ] && { echo "E5 $TAG already finished"; exit 0; }
[ -f "logs/e5_$TAG.STOP" ] && { echo "E5 $TAG stopped by user (rm logs/e5_$TAG.STOP to allow restarts)"; exit 0; }
exec 9> "logs/e5_$TAG.lock"
flock -n 9 || { echo "E5 $TAG already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }

if [ -n "${WAIT_FOR:-}" ]; then
  i=0
  until [ -f "$WAIT_FOR" ]; do
    [ $((i % 12)) -eq 0 ] && stamp "waiting for $WAIT_FOR"
    i=$((i + 1)); sleep 300
  done
fi
for i in $(seq 720); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  [ "${free:-0}" -gt 16000 ] && break
  [ $((i % 10)) -eq 1 ] && stamp "waiting for GPU memory (free=${free:-?} MiB)"
  sleep 60
done

PY=.venv/bin/python
for spec in "$@"; do
  pos=${spec%%:*}; seed=${spec##*:}
  run=parseq_${pos}_s${seed}; dir=runs/$run
  [ -f "$dir/DONE" ] && { stamp "$run already done"; continue; }
  if [ ! -f "$dir/checkpoints/final.ckpt" ]; then
    RESUME=()
    [ -f "$dir/checkpoints/last.ckpt" ] && RESUME=("ckpt_path=$PWD/$dir/checkpoints/last.ckpt")
    stamp "E5 $run on $(nvidia-smi --query-gpu=name --format=csv,noheader) ${RESUME[*]:-}"
    $PY train.py dataset=synth model.pos_enc="$pos" seed="$seed" \
        model.lr=4.9497e-4 swa_base_lr=7e-4 trainer.devices=1 +trainer.accumulate_grad_batches=2 \
        data.num_workers="${NUM_WORKERS:-12}" hydra.run.dir="$dir" "${RESUME[@]}"
  fi
  $PY eval_json.py "$dir/checkpoints/final.ckpt" --out "results/$run.json"
  touch "$dir/DONE"
  stamp "E5 $run DONE"
done
stamp "E5 $TAG ALL DONE"
touch "logs/e5_$TAG.DONE"
