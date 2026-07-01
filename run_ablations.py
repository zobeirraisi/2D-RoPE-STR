"""
Ablation driver for 2D-RoPE-STR.

Trains a set of configurations with an identical backbone/decoder and only the
positional-encoding (or 2D-RoPE hyper-parameter) varied, then prints a result
table. Each configuration is a short run controlled by --iters; use a larger
budget for paper-grade numbers.

Examples
--------
Positional-encoding ablation (Table: PE type):
    python run_ablations.py --study pe --iters 30000 --train_sources MJ --amp

2D-RoPE design ablations (dim split, frequency bases):
    python run_ablations.py --study dim  --iters 30000 --train_sources MJ --amp
    python run_ablations.py --study freq --iters 30000 --train_sources MJ --amp
"""

import argparse
import json
import os
import re

import torch

from train_2drope_str import run_training, parse_args as _train_parse


# Each study maps a label -> dict of overrides applied on top of the base args.
STUDIES = {
    "pe": {
        "None":            dict(pe_type="none"),
        "1D Sinusoidal":   dict(pe_type="1d_sin"),
        "1D Learnable":    dict(pe_type="1d_learn"),
        "2D Sinusoidal":   dict(pe_type="2d_sin"),
        "2D Learnable":    dict(pe_type="2d_learn"),
        "1D RoPE":         dict(pe_type="1drope"),
        "2D-RoPE (Ours)":  dict(pe_type="2drope", learnable_freq=True),
    },
    "dim": {
        "384:128 (75/25)": dict(pe_type="2drope", dim_split=0.75),
        "128:384 (25/75)": dict(pe_type="2drope", dim_split=0.25),
        "256:256 (50/50)": dict(pe_type="2drope", dim_split=0.5),
    },
    "freq": {
        # Matches manuscript Table (frequency base analysis). h = row, w = col.
        "Fixed theta 10000":      dict(pe_type="2drope", learnable_freq=False,
                                       rope_base_h=10000.0, rope_base_w=10000.0),
        "Fixed theta 500":        dict(pe_type="2drope", learnable_freq=False,
                                       rope_base_h=500.0, rope_base_w=500.0),
        "Learnable beta":         dict(pe_type="2drope", learnable_freq=True),
        "Mixed row500 col10000":  dict(pe_type="2drope", learnable_freq=False,
                                       rope_base_h=500.0, rope_base_w=10000.0),
    },
    "layers": {
        "3 layers":  dict(pe_type="2drope", num_encoder_layers=3),
        "6 layers":  dict(pe_type="2drope", num_encoder_layers=6),
        "9 layers":  dict(pe_type="2drope", num_encoder_layers=9),
        "12 layers": dict(pe_type="2drope", num_encoder_layers=12),
    },
    "cross": {
        # Base model is the full 2D-RoPE encoder; we vary only what is applied
        # to the decoder cross-attention keys.
        "No PE in cross-attn":   dict(pe_type="2drope", learnable_freq=True, cross_pe="none"),
        "1D RoPE in cross-attn": dict(pe_type="2drope", learnable_freq=True, cross_pe="1drope"),
        "2D-RoPE in cross-attn": dict(pe_type="2drope", learnable_freq=True, cross_pe="2drope"),
    },
}


def base_args(extra_argv):
    # Reuse the training arg parser for consistent defaults, then override.
    args = _train_parse_from(extra_argv)
    return args


def _train_parse_from(argv):
    import sys
    old = sys.argv
    sys.argv = ["train"] + argv
    try:
        return _train_parse()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=list(STUDIES.keys()))
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--train_sources", nargs="+", default=["MJ"])
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--feature_stride", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default="ablation_results.json")
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed configs (full-iter _best.pt) and resume "
                         "interrupted ones from their last snapshot")
    cli = ap.parse_args()

    shared = [
        f"--iters={cli.iters}", f"--batch_size={cli.batch_size}",
        f"--feature_stride={cli.feature_stride}",
        f"--num_workers={cli.num_workers}", "--eval_every=0",
        f"--save_dir={cli.save_dir}",
        "--train_sources", *cli.train_sources,
    ]
    if cli.limit_train:
        shared += [f"--limit_train={cli.limit_train}"]
    if cli.amp:
        shared += ["--amp"]

    def _ckpt_iter_results(path):
        if not os.path.exists(path):
            return 0, None
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("iteration", 0)), ck.get("results")

    results = {}
    for label, overrides in STUDIES[cli.study].items():
        args = base_args(shared)
        for k, v in overrides.items():
            setattr(args, k, v)
        slug = re.sub(r"[^0-9a-zA-Z]+", "-", label).strip("-").lower()
        args.run_name = f"abl_{cli.study}_{slug}"
        last_path = os.path.join(cli.save_dir, f"{args.run_name}.pt")
        best_path = os.path.join(cli.save_dir, f"{args.run_name}_best.pt")

        if cli.resume:
            _, best_res = _ckpt_iter_results(best_path)
            if best_res is not None:
                # config already ran to completion (best.pt only written at eval)
                results[label] = best_res
                print(f"[{cli.study}] {label}: complete (Avg={best_res.get('Avg', float('nan')):.2f}) -> reuse")
                continue
            snap_iter, _ = _ckpt_iter_results(last_path)
            if 0 < snap_iter < cli.iters:
                args.resume = last_path
                print(f"[{cli.study}] {label}: resuming from iter {snap_iter}")

        print("\n" + "=" * 60)
        print(f"[{cli.study}] {label}  ->  {overrides}")
        print("=" * 60)
        results[label] = run_training(args)

    print("\n\n================ ABLATION SUMMARY ================")
    cols = ["IIIT5K", "SVT", "IC13", "IC15", "CUTE80", "SVTP", "Avg"]
    header = f"{'Config':22s}" + "".join(f"{c:>9s}" for c in cols)
    print(header)
    print("-" * len(header))
    for label, res in results.items():
        row = f"{label:22s}" + "".join(f"{res.get(c, float('nan')):9.2f}" for c in cols)
        print(row)

    with open(cli.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {cli.out}")


if __name__ == "__main__":
    main()
