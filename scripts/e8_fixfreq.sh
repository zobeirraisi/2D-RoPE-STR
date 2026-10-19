#!/usr/bin/env bash
# E8 on the PC (RTX 4090): fixed-frequency 2D-RoPE (Amendment 5), seeds {0,1,2}. Same recipe as
# the E1 "pe" study (60k, stride 8, MJ+ST, bs256, AMP); only learnable_freq differs from
# "2D-RoPE (Ours)". Paired by seed with the E1 1D RoPE runs.
#
# Launch (detached):
#   cd code && setsid nohup bash scripts/e8_fixfreq.sh >> logs/e8_fixfreq.log 2>&1 < /dev/null &
# Stop:   touch logs/e8_fixfreq.STOP; pkill -f "e8_fixfre[q].sh"; pkill -f "run_ablations.py --stud[y]"
# Power-loss safe: atomic checkpoints every 2,500 iters + --resume; a crontab @reboot
# entry re-runs this script (exits if DONE/STOP or already running).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

[ -f logs/e8_fixfreq.DONE ] && { echo "E8 already finished"; exit 0; }
[ -f logs/e8_fixfreq.STOP ] && { echo "E8 stopped by user (rm logs/e8_fixfreq.STOP to allow restarts)"; exit 0; }
exec 9> logs/e8_fixfreq.lock
flock -n 9 || { echo "E8 already running"; exit 0; }

# After a reboot, wait for the data disk and the GPU driver.
for i in $(seq 60); do
  [ -d data_fast/training/MJ ] && [ -d data_fast/evaluation/SVT ] && nvidia-smi > /dev/null 2>&1 && break
  echo "waiting for data disk / GPU ($i)"; sleep 30
done

stamp() { echo "[$(date '+%F %T')] $*"; }

PY=.venv/bin/python
CK=checkpoints_rev1
RES=results/rev1
mkdir -p "$CK" "$RES"

for s in 0 1 2; do
  stamp "E8 pe_fixfreq seed $s on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  $PY run_ablations.py --study pe_fixfreq --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed "$s" --save_dir "$CK" --resume \
      --data_root data_fast --out "$RES/abl_pe_fixfreq_s$s.json"
done

stamp "E8 DONE"
touch logs/e8_fixfreq.DONE
