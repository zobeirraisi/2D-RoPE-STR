"""
Robustness to controlled geometric transformations (paper, Sec. 4.5 and Fig. 3).

Every image of the six benchmarks (7,672 in total) is transformed and the
model is re-evaluated:

  rotation     alpha in {0, 10, 20, 30, 45, 60} degrees, random sign per image,
               applied to the original crop with the canvas enlarged (no text
               is cropped), then resized to 32x128 as usual.
  perspective  keystone warp of strength s in {0, 0.1, 0.2, 0.3, 0.4}: the two
               corners of the left or right edge (chosen per image) move
               vertically towards the edge centre by s*H/2 each, applied after
               the 32x128 resize. s matches torchvision's distortion_scale
               (training augmentation uses up to 0.3, at random).

Both use fill value 0, like the training augmentation. The per-image choices
(sign, edge) depend only on (transformation seed, benchmark, image index), so
every model sees exactly the same transformed images. Level 0 is the clean
test set and is evaluated once.

Output: one JSON per checkpoint,
    results[kind][level][tseed] = {benchmark: acc, "Avg": mean of six, "All": pooled}

Usage:
    python eval_geometric.py --checkpoints checkpoints_rev1/abl_pe_*_s0.pt \
        --data_root data_fast --out_dir results/rev1/geom
"""

import argparse
import io
import json
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

from dataset import EVAL_BENCHMARKS, LabelConverter, LmdbDataset
from evaluate import evaluate_dataset
from model import STRModel

LEVELS = {
    "rotation": [0, 10, 20, 30, 45, 60],
    "perspective": [0.0, 0.1, 0.2, 0.3, 0.4],
}
IMG_H, IMG_W = 32, 128


def rotate(img, angle):
    return img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))


def keystone(img, s, edge):
    """Shrink the left or right edge of a 32x128 image vertically by s*H."""
    w, h = img.size
    d = s * h / 2
    src = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]  # TL, TR, BR, BL
    dst = [list(p) for p in src]
    if edge == "left":
        dst[0][1] += d
        dst[3][1] -= d
    else:
        dst[1][1] += d
        dst[2][1] -= d
    return TF.perspective(img, src, dst, interpolation=InterpolationMode.BILINEAR, fill=0)


class GeomDataset(LmdbDataset):
    """A benchmark LMDB with one deterministic geometric transformation."""

    def __init__(self, root, converter, bench, kind, level, tseed):
        super().__init__(root, converter, IMG_H, IMG_W, augment=False)
        self.bench, self.kind, self.level, self.tseed = bench, kind, level, tseed

    def __getitem__(self, index):
        key = self.lmdb_key(index)  # valid samples only (eval mode)
        with self.env.begin(write=False) as txn:
            label = txn.get(f"label-{key:09d}".encode()).decode("utf-8", "ignore")
            imgbuf = txn.get(f"image-{key:09d}".encode())
        img = Image.open(io.BytesIO(imgbuf)).convert("RGB")
        rng = random.Random(f"{self.tseed}/{self.bench}/{index}")
        if self.kind == "rotation" and self.level:
            img = rotate(img, self.level * rng.choice((-1, 1)))
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        if self.kind == "perspective" and self.level:
            img = keystone(img, self.level, rng.choice(("left", "right")))
        image = self.clean_transform(img)  # resize (no-op here) + normalise
        tgt_in, tgt_out = self.converter.encode(label)
        return image, tgt_in, tgt_out


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    converter = LabelConverter(max_len=cfg.get("max_len", 25))
    model = STRModel(
        vocab_size=converter.vocab_size,
        pe_type=cfg.get("pe_type", "2drope"),
        learnable_freq=cfg.get("learnable_freq", False),
        dim_split=cfg.get("dim_split", 0.5),
        feature_stride=cfg.get("feature_stride", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        rope_base_h=cfg.get("rope_base_h", 10000.0),
        rope_base_w=cfg.get("rope_base_w", 10000.0),
        cross_pe=cfg.get("cross_pe", None),
        pretrained_backbone=False,
        max_len=converter.max_len,
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device), converter, cfg, ckpt.get("results", {}), ckpt.get("iteration")


def evaluate_condition(model, converter, device, data_root, kind, level, tseed, args):
    res, correct, total = {}, 0.0, 0
    for dirname, name in EVAL_BENCHMARKS.items():
        ds = GeomDataset(os.path.join(data_root, "evaluation", dirname),
                         converter, name, kind, level, tseed)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        acc = evaluate_dataset(model, ds, converter, device,
                               batch_size=args.batch_size, num_workers=args.num_workers,
                               max_len=converter.max_len + 2, amp=True)
        res[name] = acc
        correct += acc * len(ds)
        total += len(ds)
    res["Avg"] = sum(res[n] for n in EVAL_BENCHMARKS.values()) / len(EVAL_BENCHMARKS)
    res["All"] = correct / total
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data_root", default="data_fast")
    ap.add_argument("--out_dir", default="results/rev1/geom")
    ap.add_argument("--tseeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="images per benchmark (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.checkpoints:
        run = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, f"{run}{'_limit' if args.limit else ''}.json")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        model, converter, cfg, saved, it = load_model(path, device)
        print(f"== {run} (pe={cfg.get('pe_type')}, cross={cfg.get('cross_pe')}, "
              f"seed={cfg.get('seed')}, iter={it})")

        clean = evaluate_condition(model, converter, device, args.data_root,
                                   "rotation", 0, 0, args)
        print(f"  clean           Avg {clean['Avg']:6.2f}  All {clean['All']:6.2f}")
        if not args.limit and "Avg" in saved and abs(saved["Avg"] - clean["Avg"]) > 0.05:
            print(f"  ! clean Avg differs from the training-time value {saved['Avg']:.2f}")

        results = {}
        for kind in args.kinds:
            results[kind] = {"0": {"-": clean}}
            for level in LEVELS[kind][1:]:
                results[kind][str(level)] = {}
                for ts in args.tseeds:
                    r = evaluate_condition(model, converter, device, args.data_root,
                                           kind, level, ts, args)
                    results[kind][str(level)][str(ts)] = r
                    print(f"  {kind:11s} {level:<4} t{ts}  Avg {r['Avg']:6.2f}  All {r['All']:6.2f}")

        payload = {"checkpoint": path, "run": run, "iteration": it,
                   "config": {k: cfg.get(k) for k in ("pe_type", "cross_pe", "dim_split",
                                                      "learnable_freq", "feature_stride", "seed")},
                   "levels": LEVELS, "tseeds": args.tseeds, "results": results}
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, out)
        print(f"  wrote {out}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
