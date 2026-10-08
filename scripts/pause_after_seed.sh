#!/usr/bin/env bash
# Pause the local rev1 queue as soon as a given seed's study has finished.
# No STOP marker is written, so the crontab @reboot entry resumes the queue
# (next seed) when the PC is restarted.
#
#   setsid nohup bash scripts/pause_after_seed.sh results/rev1/abl_pe_s0.json >> logs/rev1_queue.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
target="$1"

until [ -f "$target" ]; do sleep 15; done
echo "[$(date '+%F %T')] PAUSE: $target written -> stopping queue (resumes at next boot)"
pkill -f "rev1_queu[e].sh"
pkill -f "run_ablations.py --stud[y]"
pkill -f "train_2drope_st[r].py"
sleep 10
pgrep -f "run_ablations.py --stud[y]" >/dev/null && pkill -9 -f "run_ablations.py --stud[y]"
echo "[$(date '+%F %T')] PAUSED. Restart the PC (or relaunch the queue) to continue."
