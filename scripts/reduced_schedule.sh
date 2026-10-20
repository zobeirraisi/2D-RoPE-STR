#!/usr/bin/env bash
# Multi-seed controlled comparisons (60k iterations, MJ+ST, seeds 0-2):
#   pe          Table 3, seven positional encodings (stride 8, 4x16 grid)
#   factorial   Table 6, 2x2 factorial: row/column split x RoPE on cross-attention keys
#   pe_fs4      Table 5, 1D vs 2D RoPE with an 8x32 grid (stride 4, 128 x 2 accumulation steps)
#   pe_fixfreq  Table 5 (4x16 block), 2D-RoPE with fixed frequencies
# Every run resumes from its last checkpoint, so the script can be re-run after an interruption.
#
#   bash scripts/reduced_schedule.sh                     # all four studies
#   bash scripts/reduced_schedule.sh pe factorial        # a subset
# Env: PY (python), DATA (data_lmdb_release), CK (checkpoints_rev1), RES (results/rev1), SEEDS ("0 1 2")
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; DATA=${DATA:-data_lmdb_release}; CK=${CK:-checkpoints_rev1}; RES=${RES:-results/rev1}
mkdir -p "$CK" "$RES"
STUDIES=("$@"); [ ${#STUDIES[@]} -eq 0 ] && STUDIES=(pe factorial pe_fs4 pe_fixfreq)

for study in "${STUDIES[@]}"; do
  case $study in
    pe_fs4) sched=(--feature_stride 4 --batch_size 128 --accum_steps 2) ;;
    pe|factorial|pe_fixfreq) sched=(--feature_stride 8) ;;
    *) echo "unknown study: $study"; exit 1 ;;
  esac
  for s in ${SEEDS:-0 1 2}; do
    $PY run_ablations.py --study "$study" --iters 60000 "${sched[@]}" --amp \
        --train_sources MJ ST --seed "$s" --save_dir "$CK" --resume \
        --data_root "$DATA" --out "$RES/abl_${study}_s$s.json"
  done
done
