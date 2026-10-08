#!/usr/bin/env bash
# IJDAR revision-1 experiment queue (runs sequentially on the local RTX 4090).
# Safe to re-run: every step resumes / skips completed work.
#
#   E1  PE ablation, 7 encodings x seeds {0,1,2}        (60k, stride 8, MJ+ST)
#   E3  full-schedule 1D RoPE baseline, seed 0         (300k, stride 4, final ckpt)
#   E2  2x2 factorial (split x cross keys) x seeds {0,1,2}
#
# Launch (detached, survives logout):
#   cd ~/Documents/RESEARCH/ROPE/code
#   setsid nohup bash scripts/rev1_queue.sh >> logs/rev1_queue.log 2>&1 < /dev/null &
# Monitor:  tail -f logs/rev1_queue.log
# Stop:     touch logs/rev1_queue.STOP; pkill -f rev1_queue.sh; pkill -f run_ablations.py; pkill -f train_2drope_str.py
#
# Power-loss safety: checkpoints are written atomically every 2,500 iters and
# every step below resumes from them. A crontab @reboot entry re-runs this
# script after a reboot; it exits at once if the queue is finished (DONE
# marker), deliberately stopped (STOP marker), or already running (flock).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f logs/rev1_queue.DONE ] && { echo "queue already finished"; exit 0; }
[ -f logs/rev1_queue.STOP ] && { echo "queue stopped by user (rm logs/rev1_queue.STOP to allow restarts)"; exit 0; }
exec 9> logs/rev1_queue.lock
flock -n 9 || { echo "queue already running"; exit 0; }

# After a reboot, wait for the data disk (data_lmdb_release is a symlink to it)
# and the GPU driver.
for i in $(seq 60); do
  [ -d data_fast/training/MJ ] && [ -d data_fast/evaluation/SVT ] && nvidia-smi > /dev/null 2>&1 && break
  echo "waiting for data disk / GPU ($i)"; sleep 30
done

PY=.venv/bin/python
DATA=data_fast              # NVMe copy of training LMDBs (the /mnt HDD starves the GPU);
                            # evaluation/ is a symlink to the original on /mnt
CK=checkpoints_rev1          # on the root disk; the data disk is 97% full
RES=results/rev1
mkdir -p "$CK" "$RES"

stamp() { echo "[$(date '+%F %T')] $*"; }

# ---------- E1: positional-encoding ablation, 3 seeds ----------
for s in 0 1 2; do
  stamp "E1 pe seed $s"
  $PY run_ablations.py --study pe --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed "$s" --save_dir "$CK" --resume \
      --data_root "$DATA" --out "$RES/abl_pe_s$s.json"
done

# ---------- E3: full-schedule 1D RoPE (same recipe as the Table-2 model) ----------
# Skipped here if it was moved to the 3090 (marker created when that run starts).
RUN=full_1drope_s0
if [ -f logs/E3_ON_3090 ]; then
  stamp "E3 skipped: running on gpu3090 (~/ROPE_rev1/code)"
else
  RESUME=()
  [ -f "$CK/$RUN.pt" ] && RESUME=(--resume "$CK/$RUN.pt")
  stamp "E3 $RUN ${RESUME[*]:-}"
  $PY train_2drope_str.py --pe_type 1drope --train_sources MJ ST --iters 300000 \
      --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 \
      --eval_every 0 --save_dir "$CK" --run_name "$RUN" --data_root "$DATA" "${RESUME[@]}"
  $PY evaluate.py --checkpoint "$CK/$RUN.pt" --data_root "$DATA" --out "$RES/$RUN.json"
fi

# ---------- E2: 2x2 factorial, 3 seeds ----------
for s in 0 1 2; do
  stamp "E2 factorial seed $s"
  $PY run_ablations.py --study factorial --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed "$s" --save_dir "$CK" --resume \
      --data_root "$DATA" --out "$RES/abl_factorial_s$s.json"
done

stamp "ALL DONE"
touch logs/rev1_queue.DONE
