#!/usr/bin/env bash
# Table 7: PARSeq with its learned additive position embedding vs. 2D-RoPE in the ViT encoder,
# everything else unchanged. Needs a PARSeq checkout at commit 1902db0 with
# patches/parseq_rope2d.patch applied (see README) and the MJ+ST LMDBs under its data/ directory.
# The official synthetic-data recipe (20 epochs, batch 384, 2 GPUs) is reproduced on ONE GPU:
# 2-step gradient accumulation (effective batch 768) and model.lr rescaled so the optimizer lr
# equals the 2-GPU value (7e-4*sqrt(2)*384/256); swa_base_lr keeps the official SWA lr.
# Each run is evaluated from final.ckpt (end of training). An interrupted run resumes from
# last.ckpt; finished runs are skipped.
#
#   bash scripts/parseq_transfer.sh rope2d:0 learned:0 rope2d:1 learned:1 rope2d:2 learned:2
# Env: PARSEQ_DIR (../third_party/parseq), PY (python), NUM_WORKERS (12)
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PARSEQ_DIR:-$CODE/../third_party/parseq}"
PY=${PY:-python}
mkdir -p runs results

for spec in "$@"; do
  pos=${spec%%:*}; seed=${spec##*:}
  run=parseq_${pos}_s${seed}; dir=runs/$run
  [ -f "results/$run.json" ] && { echo "$run: done"; continue; }
  if [ ! -f "$dir/checkpoints/final.ckpt" ]; then
    resume=()
    [ -f "$dir/checkpoints/last.ckpt" ] && resume=("ckpt_path=$PWD/$dir/checkpoints/last.ckpt")
    $PY train.py dataset=synth model.pos_enc="$pos" seed="$seed" \
        model.lr=4.9497e-4 swa_base_lr=7e-4 trainer.devices=1 +trainer.accumulate_grad_batches=2 \
        data.num_workers="${NUM_WORKERS:-12}" hydra.run.dir="$dir" "${resume[@]}"
  fi
  $PY eval_json.py "$dir/checkpoints/final.ckpt" --out "results/$run.json"
done
