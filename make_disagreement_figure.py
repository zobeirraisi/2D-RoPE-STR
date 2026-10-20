"""
Fig. 4 of the paper: disagreements between the single-run 1D-sinusoidal and 2D-RoPE
ablation models (the pair of Table 8), in BOTH directions, chosen by a rule fixed before any
per-image prediction was inspected:

  for each of CUTE80, SVTP and IC15 (2,077) and each direction (only 2D-RoPE correct / only
  1D sinusoidal correct), the first image in LMDB order whose normalized label has >= 5 characters.

Correctness and the printed predictions are the batched outputs behind Table 8 (same loader and
decoding as `make_qualitative.per_sample_correct`). Before drawing, both models' accuracies are
recomputed on all six benchmarks and must equal their single-run entries in results/abl_pe.json
(Table 3, left). The disagreement counts are printed next to results/netwins.json (Table 8).

Run from code/:
  python make_disagreement_figure.py [--out ../manuscript/ijdar/revision1/fig_qualitative.pdf]
"""
import argparse
import io
import json
import os

import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import EVAL_BENCHMARKS, LmdbDataset
from evaluate import normalize
import make_qualitative as mq

ROOT = "data_lmdb_release/evaluation"
PAIR = {"2D-RoPE": "checkpoints/abl_pe_2d-rope-ours_best.pt",
        "1D sin.": "checkpoints/abl_pe_1d-sinusoidal_best.pt"}
SHOWN = [("CUTE80", "CUTE80"), ("SVTP", "SVTP"), ("IC15_2077", "IC15")]
MIN_LEN = 5
OK, BAD, INK, MUTED = "#1a7f37", "#c62828", "#222222", "#666666"


@torch.no_grad()
def per_sample_preds(model, conv, ds):
    """(prediction, ground truth) per sample, in dataset order; same loader and decoding as
    make_qualitative.per_sample_correct, so correctness matches Table 8."""
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    out = []
    for images, _, tgt_out in loader:
        images = images.to(mq.DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=mq.DEVICE.type == "cuda"):
            pred = model.greedy_decode(images, conv.SOS, conv.EOS, conv.max_len + 2)
        for p, gt in zip(pred.cpu(), tgt_out):
            out.append((conv.decode(p.tolist()), conv.decode(gt.tolist())))
    return out


def raw_image(path, key):
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        img = Image.open(io.BytesIO(txn.get(f"image-{key:09d}".encode()))).convert("RGB")
    env.close()
    return img


def select(models, conv, netwins, acc_ref):
    picks = []
    for dirname, name in EVAL_BENCHMARKS.items():
        path = os.path.join(ROOT, dirname)
        ds = LmdbDataset(path, conv)
        p2, p1 = per_sample_preds(models["2D-RoPE"], conv, ds), per_sample_preds(models["1D sin."], conv, ds)
        c2 = np.array([normalize(p) == normalize(g) for p, g in p2])
        c1 = np.array([normalize(p) == normalize(g) for p, g in p1])
        for acc, key in ((c2, "2D-RoPE (Ours)"), (c1, "1D Sinusoidal")):
            assert abs(100 * acc.mean() - acc_ref[key][name]) < 1e-9, f"{name} {key}: accuracy differs from abl_pe.json"
        got = {"2drope_only": int((c2 & ~c1).sum()), "1dsin_only": int((c1 & ~c2).sum())}
        ref = {k: netwins[name][k] for k in got}
        print(f"  {name:7s} accuracies = abl_pe.json; 2D-only={got['2drope_only']:3d} 1D-only={got['1dsin_only']:3d}"
              f"  (netwins.json: {ref['2drope_only']}, {ref['1dsin_only']}{'' if got == ref else '  <- differs'})")
        if dirname not in dict(SHOWN):
            continue
        for direction, mask in (("2D-RoPE only", c2 & ~c1), ("1D sin. only", c1 & ~c2)):
            i = next(i for i in np.flatnonzero(mask) if len(normalize(p2[i][1])) >= MIN_LEN)
            key = ds.keys[i]
            picks.append({"benchmark": dict(SHOWN)[dirname], "direction": direction, "lmdb_key": int(key),
                          "gt": normalize(p2[i][1]), "1D sin.": normalize(p1[i][0]),
                          "2D-RoPE": normalize(p2[i][0]), "image": raw_image(path, key)})
            print(f"    {direction:13s} {dict(SHOWN)[dirname]:6s} #{key}: GT={picks[-1]['gt']!r} "
                  f"1D={picks[-1]['1D sin.']!r} 2D={picks[-1]['2D-RoPE']!r}")
    return picks


def render(picks, out):
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    W, left, gap, cw = 372 / 72.27, 0.95, 0.10, 1.32          # inches; printed at text width
    ih, th, row_h, top = cw / 4, 0.12, 0.95, 0.18             # image 4:1 like the model input
    H = top + row_h + ih + 0.03 + 3 * th + 0.04               # two rows; last row's text + margin
    fig = plt.figure(figsize=(W, H))
    rows = ["2D-RoPE only", "1D sin. only"]
    row_text = {"2D-RoPE only": "only\n2D-RoPE\ncorrect", "1D sin. only": "only\n1D sinusoidal\ncorrect"}
    cols = [b for _, b in SHOWN]
    for r, direction in enumerate(rows):
        y_img = H - top - r * row_h - ih                     # bottom of the image row
        fig.text(0.02 / W, (y_img + ih / 2) / H, row_text[direction], ha="left", va="center",
                 fontsize=7.5, color=INK, linespacing=1.2)
        for c, bench in enumerate(cols):
            p = next(q for q in picks if q["benchmark"] == bench and q["direction"] == direction)
            x0 = left + c * (cw + gap)
            ax = fig.add_axes([x0 / W, y_img / H, cw / W, ih / H])
            ax.imshow(p["image"].resize((128, 32)), extent=[0, 128, 32, 0])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color(MUTED)
            ax.set_title(bench, fontsize=7.5, color=MUTED, pad=2)
            lines = [(f"GT: {p['gt']}", INK)]
            for m in ("1D sin.", "2D-RoPE"):
                ok = p[m] == p["gt"]
                lines.append((f"{m:8s} {p[m] or '(empty)'} {'✓' if ok else '✗'}", OK if ok else BAD))
            for k, (s, col) in enumerate(lines):
                fig.text((x0 + 0.02) / W, (y_img - 0.03 - (k + 1) * th) / H, s, ha="left", va="bottom",
                         fontsize=7, family="monospace", color=col)
    fig.savefig(out)
    fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=250)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../manuscript/ijdar/revision1/fig_qualitative.pdf")
    ap.add_argument("--picks", default="results/rev1/fig4_disagreements.json")
    args = ap.parse_args()
    models = {}
    for label, path in PAIR.items():
        models[label], conv = mq.load_model(path)
        print(f"loaded {label}: {path}")
    picks = select(models, conv, json.load(open("results/netwins.json")), json.load(open("results/abl_pe.json")))
    json.dump([{k: v for k, v in p.items() if k != "image"} for p in picks], open(args.picks, "w"), indent=1)
    print(f"wrote {args.picks}")
    render(picks, args.out)


if __name__ == "__main__":
    main()
