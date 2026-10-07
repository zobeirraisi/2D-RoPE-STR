"""
Side-by-side encoder self-attention for several positional encodings on the SAME
images (IJDAR revision, Reviewer 2 comment 4).

Examples are chosen without looking at the predictions: the first N distinct
words of >= MIN_LEN characters in each benchmark, in LMDB order. Each model's
prediction is printed under its panel (green = correct, red = wrong), and every
row shares one colour scale, so the panels are directly comparable.

Saliency = encoder self-attention received by each grid cell, averaged over
queries, heads and encoder layers (same definition as make_qualitative.py).

Run from code/:
  python make_attention_sbs.py \
      --models "1D sinusoidal=checkpoints/abl_pe_1d-sinusoidal_best.pt" \
               "1D RoPE=checkpoints/abl_pe_1d-rope_best.pt" \
               "2D-RoPE=checkpoints/abl_pe_2d-rope-ours_best.pt" \
      --out ../manuscript/ijdar/revision1/fig_attention_sbs.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import LabelConverter, build_transform
from evaluate import normalize
from model import STRModel
import make_qualitative as mq

ROOT = "data_lmdb_release/evaluation"


def load_model(path):
    ck = torch.load(path, map_location=mq.DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    conv = LabelConverter(max_len=cfg.get("max_len", 25))
    m = STRModel(
        vocab_size=conv.vocab_size, pe_type=cfg.get("pe_type", "2drope"),
        learnable_freq=cfg.get("learnable_freq", False),
        dim_split=cfg.get("dim_split", 0.5),
        feature_stride=cfg.get("feature_stride", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        rope_base_h=cfg.get("rope_base_h") or 10000.0,
        rope_base_w=cfg.get("rope_base_w") or 10000.0,
        cross_pe=cfg.get("cross_pe"), pretrained_backbone=False,
        max_len=conv.max_len,
    ).to(mq.DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, conv


def pick_examples(conv, benchmarks, per_set, min_len):
    out = []
    for name in benchmarks:
        seen = set()
        for raw, tensor, gt in mq.lmdb_samples(f"{ROOT}/{name}", conv, limit=400):
            g = normalize(gt)
            if len(g) >= min_len and g not in seen:
                out.append((raw, tensor, gt, name))
                seen.add(g)
                if len(seen) >= per_set:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help='"Label=path/to/ckpt.pt" entries, shown left to right')
    ap.add_argument("--benchmarks", nargs="+",
                    default=["CUTE80", "SVTP", "IC15_2077"])
    ap.add_argument("--per_set", type=int, default=2)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = []
    for spec in args.models:
        label, path = spec.split("=", 1)
        m, conv = load_model(path)
        models.append((label, m))
        print(f"loaded {label}: {path}")

    examples = pick_examples(conv, args.benchmarks, args.per_set, args.min_len)
    tf = build_transform(augment=False)
    nr, nc = len(examples), len(models) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(1.75 * nc, 0.78 * nr))
    axes = np.array(axes).reshape(nr, nc)

    for r, (raw, _, gt, bench) in enumerate(examples):
        x = tf(raw)
        sals = [mq.encoder_saliency(m, conv, x)[0] for _, m in models]
        ups = [np.kron(s, np.ones((32 // s.shape[0], 128 // s.shape[1]))) for s in sals]
        vrange = (min(u.min() for u in ups), max(u.max() for u in ups))  # shared per row

        mq._overlay(axes[r, 0], raw, None)
        axes[r, 0].set_xlabel(f"GT: {normalize(gt)}", fontsize=6.5,
                              family="monospace", labelpad=1)
        axes[r, 0].set_ylabel(bench.split("_")[0], fontsize=6.5, rotation=0,
                              ha="right", va="center")
        for c, ((label, m), s) in enumerate(zip(models, sals), start=1):
            ax = axes[r, c]
            mq._overlay(ax, raw, s, vrange)
            pred = normalize(mq.predict(m, conv, x))
            ok = pred == normalize(gt)
            ax.set_xlabel(pred or "(empty)", fontsize=6.5, family="monospace",
                          color="green" if ok else "red", labelpad=1)
            if r == 0:
                ax.set_title(label, fontsize=7.5)
        if r == 0:
            axes[r, 0].set_title("input", fontsize=7.5)
        print(f"  row {r}: {bench} GT={gt!r}")

    fig.tight_layout(h_pad=0.25, w_pad=0.3)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
