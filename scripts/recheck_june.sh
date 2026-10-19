#!/usr/bin/env bash
# Re-evaluate the June single-run checkpoints (Table 3 single runs, Table 4, main model of Table 2)
# and compare per-benchmark correct counts with the stored JSONs and with the results saved inside
# each checkpoint. Motivated by the stale results/netwins.json found on 2026-10-19. Inference only.
# Waits for E8 (logs/e8_fixfreq.DONE) so it never shares the GPU with a training run.
#
# Launch (detached):
#   cd code && setsid nohup bash scripts/recheck_june.sh >> logs/recheck_june.log 2>&1 < /dev/null &
# Resumable: finished checkpoints are skipped. Output: results/rev1/recheck/{<run>.json,summary.txt}
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/rev1/recheck

[ -f logs/recheck_june.DONE ] && { echo "recheck already finished"; exit 0; }
exec 9> logs/recheck_june.lock
flock -n 9 || { echo "recheck already running"; exit 0; }

stamp() { echo "[$(date '+%F %T')] $*"; }

until [ -f logs/e8_fixfreq.DONE ]; do
  [ -f logs/e8_fixfreq.STOP ] && { stamp "E8 stopped by user; recheck not started"; exit 0; }
  sleep 120
done
sleep 60   # let E8 release the GPU

PY=.venv/bin/python
OUT=results/rev1/recheck
for ck in checkpoints/abl_*_best.pt checkpoints/full_2drope_best.pt; do
  run=$(basename "$ck" _best.pt)
  [ -f "$OUT/$run.json" ] && continue
  stamp "evaluating $run"
  $PY evaluate.py --checkpoint "$ck" --data_root data_lmdb_release --batch_size 256 \
      --out "$OUT/$run.json.tmp"
  mv "$OUT/$run.json.tmp" "$OUT/$run.json"
done

$PY - <<'EOF' | tee "$OUT/summary.txt"
import glob, json, os, re
import torch

N = {"IIIT5K": 3000, "SVT": 647, "IC13": 1015, "IC15": 2077, "CUTE80": 288, "SVTP": 645}
slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
stored = {}
for f in ("abl_pe", "abl_dim", "abl_freq", "abl_layers", "abl_cross"):
    for k, v in json.load(open(f"results/{f}.json")).items():
        stored[f"{f}_{slug(k)}"] = (f"{f}.json[{k}]", v)
stored["full_2drope"] = ("full_2drope.json", json.load(open("results/full_2drope.json")))
count = lambda r: {b: round(r[b] * n / 100) for b, n in N.items()}

n_bad = 0
for path in sorted(glob.glob("results/rev1/recheck/*.json")):
    run = os.path.basename(path)[:-5]
    new = count(json.load(open(path)))
    ck = torch.load(f"checkpoints/{run}_best.pt", map_location="cpu", mmap=True, weights_only=False)
    src, old = stored.get(run, ("(no JSON entry)", None))
    diffs = []
    for name, ref in (("ckpt", ck["results"]), (src, old)):
        if ref is None:
            diffs.append(f"{name}: missing"); continue
        d = {b: new[b] - c for b, c in count(ref).items() if new[b] != c}
        if d:
            diffs.append(f"{name}: {d}")
    n_bad += bool(diffs)
    print(f"{run:40s} iter {ck['iteration']:>6}  " + ("OK" if not diffs else "DIFF  " + "; ".join(diffs)))
print(f"\n{n_bad} of {len(glob.glob('results/rev1/recheck/*.json'))} runs differ "
      "(re-evaluated minus stored correct counts per benchmark)")
EOF

stamp "recheck DONE"
touch logs/recheck_june.DONE
