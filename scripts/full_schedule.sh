#!/usr/bin/env bash
# Full-schedule rows of Table 2: seeded 2D-RoPE (main configuration), 1D RoPE, and 1D sinusoidal,
# 300k iterations on MJ+ST, stride 4 (8x32 grid), 128 x 2 accumulation steps. The final-iteration
# checkpoint of each run is evaluated. Re-running resumes an interrupted run and skips finished ones.
#
#   bash scripts/full_schedule.sh
# Env: PY (python), DATA (data_lmdb_release), CK (checkpoints_rev1), RES (results/rev1)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; DATA=${DATA:-data_lmdb_release}; CK=${CK:-checkpoints_rev1}; RES=${RES:-results/rev1}
mkdir -p "$CK" "$RES"

run() {  # run_name, then the encoding arguments
  local name=$1; shift
  [ -f "$RES/$name.json" ] && { echo "$name: done"; return; }
  local resume=()
  [ -f "$CK/$name.pt" ] && resume=(--resume "$CK/$name.pt")
  $PY train_2drope_str.py "$@" --train_sources MJ ST --iters 300000 \
      --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 \
      --eval_every 0 --save_dir "$CK" --run_name "$name" --data_root "$DATA" "${resume[@]}"
  $PY evaluate.py --checkpoint "$CK/$name.pt" --data_root "$DATA" --out "$RES/$name.json"
}

run full_2drope_s0 --pe_type 2drope --learnable_freq --dim_split 0.5
run full_1drope_s0 --pe_type 1drope
run full_1dsin_s0  --pe_type 1d_sin
