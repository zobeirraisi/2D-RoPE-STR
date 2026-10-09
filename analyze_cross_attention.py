"""
Cross-attention diagnostic for the factorial / cross-attention ablations (IJDAR revision, R1-3).

Proposition 1 says that rotating only the encoder keys gives each decoder query a way to
address *absolute* 2D locations. With a 2D-RoPE encoder (relative only), the memory has no
explicit absolute position otherwise, so the prediction is that rotated keys help the decoder
keep its place in the reading order. We test this on the decoder cross-attention maps, under
teacher forcing on the ground-truth string, for every test image:

  spread       attention-weighted std of the attended grid position (in grid cells),
               averaged over character steps
  entropy      normalised entropy of the attention distribution (0 = one cell, 1 = uniform)
  backtrack    fraction of consecutive character steps whose attention centroid moves
               left by more than half a cell (a reading-order violation)
  backtrack_peak  the same with the column of the attention peak (robust to diffuse maps)

Attention is averaged over heads; metrics are stored for every decoder layer (L1..L6), the last
layer, and the mean over layers. Report the full per-layer profile, not a selected layer. Per benchmark, and pooled over regular (IIIT5K, SVT, IC13) and irregular
(IC15, CUTE80, SVTP) images, also split by whether greedy decoding reads the word correctly.
Only models with the custom cross-attention decoder (config cross_pe set) are supported.

Usage:
    python analyze_cross_attention.py --checkpoints checkpoints_rev1/abl_factorial_*_s[0-9].pt \
        --data_root data_fast --out_dir results/rev1/xattn
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model as model_mod
from dataset import EVAL_BENCHMARKS, LmdbDataset
from eval_geometric import load_model
from evaluate import normalize

METRICS = ("spread", "entropy", "backtrack", "backtrack_peak")
REGULAR = ("IIIT5K", "SVT", "IC13")
IRREGULAR = ("IC15", "CUTE80", "SVTP")


def _cross_attn_with_weights(self, x, memory, H, W):
    """Same computation as CrossRoPEDecoderLayer._cross_attn, keeping the attention weights."""
    B, T, D = x.shape
    S = memory.size(1)
    q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    kv = self.kv_proj(memory).view(B, S, 2, self.n_heads, self.head_dim)
    k, v = kv.permute(2, 0, 3, 1, 4)
    if self.cross_pe == "2drope":
        hf, wf = self.rope(torch.arange(H, device=x.device), torch.arange(W, device=x.device))
        k = model_mod.apply_2d_rope(k, hf, wf, H, W, self.dim_split)
    elif self.cross_pe == "1drope":
        k = model_mod.apply_1d_rope(k, self.rope(S, x.device))
    attn = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
    self.last_attn = attn.mean(1)  # (B, T, S), averaged over heads
    out = (attn.to(v.dtype) @ v)
    return self.cross_out(out.transpose(1, 2).contiguous().view(B, T, D))


def step_metrics(attn, H, W, lengths):
    """attn: (B, T, S) for decoder inputs [SOS, c1..]; step t predicts character t+1.
    Returns per-image sums over character steps and counts."""
    B, T, S = attn.shape
    dev = attn.device
    hh = torch.arange(H, device=dev).repeat_interleave(W).float()   # row of each cell
    ww = torch.arange(W, device=dev).repeat(H).float()              # column of each cell
    ch = attn @ hh                                                  # (B, T) centroid row
    cw = attn @ ww
    var = attn @ (hh ** 2) - ch ** 2 + attn @ (ww ** 2) - cw ** 2
    spread = var.clamp_min(0).sqrt()
    ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1) / math.log(S)
    steps = torch.arange(T, device=dev)[None, :] < lengths[:, None]  # character steps only
    back = (cw[:, 1:] < cw[:, :-1] - 0.5) & steps[:, 1:]
    pw = (attn.argmax(-1) % W).float()                              # column of the attention peak
    back_peak = (pw[:, 1:] < pw[:, :-1]) & steps[:, 1:]
    return {
        "spread": (spread * steps).sum(1), "entropy": (ent * steps).sum(1), "n_steps": steps.sum(1),
        "backtrack": back.sum(1).float(), "backtrack_peak": back_peak.sum(1).float(),
        "n_pairs": steps[:, 1:].sum(1),
    }


@torch.no_grad()
def analyse(model, converter, device, data_root, args):
    for layer in model.cross_decoder:
        layer._cross_attn = _cross_attn_with_weights.__get__(layer)
    per_bench = {}
    for dirname, name in EVAL_BENCHMARKS.items():
        ds = LmdbDataset(os.path.join(data_root, "evaluation", dirname), converter)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
        acc = {"correct": [], "n_steps": [], "n_pairs": []}
        for images, tgt_in, tgt_out in loader:
            images, tgt_in = images.to(device), tgt_in.to(device)
            lengths = (tgt_out != converter.PAD).sum(1).to(device) - 1  # characters, without EOS
            pred = model.greedy_decode(images, converter.SOS, converter.EOS, converter.max_len + 2)
            acc["correct"] += [normalize(converter.decode(p.tolist())) == normalize(converter.decode(g.tolist()))
                               for p, g in zip(pred.cpu(), tgt_out)]
            model(images, tgt_in)  # teacher forcing; fills layer.last_attn
            H, W = model._mem_hw
            per_layer = [step_metrics(l.last_attn, H, W, lengths) for l in model.cross_decoder]
            last = per_layer[-1]
            for key in METRICS:
                for i, m in enumerate(per_layer):
                    acc.setdefault(f"L{i + 1}_{key}", []).extend(m[key].tolist())
                acc.setdefault(f"last_{key}", []).extend(last[key].tolist())
                acc.setdefault(f"mean_{key}", []).extend(torch.stack([m[key] for m in per_layer]).mean(0).tolist())
            acc["n_steps"] += last["n_steps"].tolist()
            acc["n_pairs"] += last["n_pairs"].tolist()
        per_bench[name] = acc
    return per_bench


def summarise(per_bench, names, which=None):
    """Pool images of the given benchmarks; which in {None, True, False} filters by correctness."""
    out = {}
    keys = [k for k in per_bench[names[0]] if k not in ("correct", "n_steps", "n_pairs")]
    for key in keys:
        num = den = 0.0
        for n in names:
            b = per_bench[n]
            norm = "n_pairs" if "backtrack" in key else "n_steps"
            for v, c, d in zip(b[key], b["correct"], b[norm]):
                if which is None or c == which:
                    num += v
                    den += d
        out[key] = num / den if den else float("nan")
    n_img = sum(1 for n in names for c in per_bench[n]["correct"] if which is None or c == which)
    out["n_images"] = n_img
    out["accuracy"] = 100.0 * sum(c for n in names for c in per_bench[n]["correct"]) / \
        sum(len(per_bench[n]["correct"]) for n in names)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--data_root", default="data_fast")
    ap.add_argument("--out_dir", default="results/rev1/xattn")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.checkpoints:
        run = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, f"{run}.json")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        model, converter, cfg, _, it = load_model(path, device)
        if model.cross_decoder is None:
            print(f"[skip] {run}: legacy decoder (cross_pe=None), no cross-attention hook")
            continue
        model.eval()
        per_bench = analyse(model, converter, device, args.data_root, args)
        groups = {**{n: (n,) for n in EVAL_BENCHMARKS.values()},
                  "regular": REGULAR, "irregular": IRREGULAR, "all": REGULAR + IRREGULAR}
        summary = {g: {"all": summarise(per_bench, names),
                       "correct": summarise(per_bench, names, True),
                       "wrong": summarise(per_bench, names, False)} for g, names in groups.items()}
        payload = {"checkpoint": path, "run": run, "iteration": it,
                   "config": {k: cfg.get(k) for k in ("pe_type", "cross_pe", "dim_split", "seed")},
                   "summary": summary}
        with open(out + ".tmp", "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(out + ".tmp", out)
        s = summary
        print(f"== {run} (cross={cfg.get('cross_pe')}, split={cfg.get('dim_split')}, seed={cfg.get('seed')})")
        for g in ("regular", "irregular", "all"):
            a = s[g]["all"]
            print(f"  {g:9s} acc {a['accuracy']:6.2f}  spread {a['last_spread']:.3f}  entropy "
                  f"{a['last_entropy']:.3f}  backtrack {100 * a['last_backtrack']:.2f}%  "
                  f"(wrong imgs: backtrack {100 * s[g]['wrong']['last_backtrack']:.2f}%, "
                  f"correct: {100 * s[g]['correct']['last_backtrack']:.2f}%)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
