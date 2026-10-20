#!/usr/bin/env bash
# Evaluations of trained models: controlled rotation / perspective (Fig. 3) for the Table 3 and
# factorial models and the four full-schedule checkpoints, and the per-layer cross-attention
# diagnostic of the factorial models (Sec. 4.4.4). Needs the checkpoints written by
# reduced_schedule.sh and full_schedule.sh (or the released ones, see README).
#
#   bash scripts/diagnostics.sh
# Env: PY (python), DATA (data_lmdb_release), CK (checkpoints_rev1), RES (results/rev1)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}; DATA=${DATA:-data_lmdb_release}; CK=${CK:-checkpoints_rev1}; RES=${RES:-results/rev1}

$PY eval_geometric.py --data_root "$DATA" --out_dir "$RES/geom" \
    --checkpoints "$CK"/abl_pe_*_s[0-9].pt "$CK"/abl_factorial_*_s[0-9].pt
$PY eval_geometric.py --data_root "$DATA" --out_dir "$RES/geom" \
    --checkpoints "$CK"/full_1drope_s0.pt "$CK"/full_1dsin_s0.pt "$CK"/full_2drope_s0.pt \
                  checkpoints/full_2drope_best.pt
$PY analyze_cross_attention.py --data_root "$DATA" --out_dir "$RES/xattn" \
    --checkpoints "$CK"/abl_factorial_*_s[0-9].pt
