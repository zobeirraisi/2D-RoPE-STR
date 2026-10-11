#!/usr/bin/env bash
# E7 on the RTX 3090 server: 1D RoPE vs 2D-RoPE with 8 feature rows (stride 4)
# instead of the 4 rows (stride 8) of E1. Pre-specified hypothesis in
# manuscript/ijdar/revision1/REVISION_NOTES.md ("E7"). 60k iters, MJ+ST,
# effective batch 256 (128 x accum 2, as E3), final checkpoint. THIS SCRIPT: seed 2 only
# (Amendment 3 in REVISION_NOTES; seeds 0-1 ran in e7_fs4_3090.sh).
# Lives in ~/ROPE_rev1/code on gpu3090; starts after the 3090 E5 seed-1 pair
# (../third_party/parseq/logs/e5_3090.DONE).
#
# Launch (detached):
#   cd ~/ROPE_rev1/code && setsid nohup bash scripts/e7s2_3090.sh >> logs/e7s2_3090.log 2>&1 < /dev/null &
# Stop:   touch logs/e7s2_3090.STOP; pkill -f "e7s2_309[0].sh"; pkill -f "run_ablations.py --stud[y]"
# Power-loss safe: atomic checkpoints every 2,500 iters + run_ablations --resume;
# a crontab @reboot entry re-runs this script (exits if DONE/STOP or already running).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

[ -f logs/e7s2_3090.DONE ] && { echo "E7 seed 2 already finished"; exit 0; }
[ -f logs/e7s2_3090.STOP ] && { echo "E7 seed 2 stopped by user (rm logs/e7s2_3090.STOP to allow restarts)"; exit 0; }
exec 9> logs/e7s2_3090.lock
flock -n 9 || { echo "E7 seed 2 already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }

i=0
WAIT_FOR="${WAIT_FOR:-$PWD/../third_party/parseq/logs/e5_3090.DONE}"
until [ -f "$WAIT_FOR" ]; do
  [ $((i % 12)) -eq 0 ] && stamp "waiting for $WAIT_FOR"
  i=$((i + 1)); sleep 300
done

# Wait until the GPU has room — needs ~12 GB.
for i in $(seq 720); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
  [ "${free:-0}" -gt 14000 ] && break
  [ $((i % 10)) -eq 1 ] && stamp "waiting for GPU memory (free=${free:-?} MiB)"
  sleep 60
done

PY=.venv/bin/python
CK=checkpoints_rev1
RES=results/rev1
mkdir -p "$CK" "$RES"

for s in 2; do
  stamp "E7 pe_fs4 seed $s on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  $PY run_ablations.py --study pe_fs4 --iters 60000 --feature_stride 4 \
      --batch_size 128 --accum_steps 2 --amp --train_sources MJ ST --seed "$s" \
      --save_dir "$CK" --resume --data_root data --out "$RES/abl_pe_fs4_s$s.json"
done
stamp "E7 seed 2 DONE"
touch logs/e7s2_3090.DONE
