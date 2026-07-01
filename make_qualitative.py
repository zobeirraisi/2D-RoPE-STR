"""
Generate REAL qualitative + ablation figures for the manuscript from the
trained ablation checkpoints (no fabrication).

  fig_qualitative.png : real eval crops where 2D-RoPE reads the word correctly
                        and the 1D-sinusoidal baseline does not. Both models are
                        the PE-ablation pair (identical schedule), so the
                        comparison is apples-to-apples.
  fig_ablation.png    : (a) PE-type Avg accuracy bar chart, (b) encoder-depth
                        accuracy curve, from code/results/*.json.

Run from code/:  python make_qualitative.py
"""
import io
import json
import math
import os

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import LabelConverter, LmdbDataset, EVAL_BENCHMARKS, build_transform
from model import STRModel
from evaluate import normalize

OUT_DIR = "../manuscript/figures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    conv = LabelConverter(max_len=cfg.get("max_len", 25))
    model = STRModel(
        vocab_size=conv.vocab_size,
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
        max_len=conv.max_len,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, conv


def lmdb_samples(path, conv, limit=None):
    """Yield (raw_PIL_rgb, input_tensor, gt_string) for valid samples."""
    tf = build_transform(augment=False)
    env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        n = int(txn.get(b"num-samples"))
        for i in range(1, n + 1):
            if limit and i > limit:
                break
            label = txn.get(f"label-{i:09d}".encode())
            buf = txn.get(f"image-{i:09d}".encode())
            if label is None or buf is None:
                continue
            gt = label.decode("utf-8", "ignore")
            if not (0 < len(conv.clean(gt)) <= conv.max_len):
                continue
            try:
                img = Image.open(io.BytesIO(buf)).convert("RGB")
            except Exception:
                continue
            yield img, tf(img), gt
    env.close()


@torch.no_grad()
def predict(model, conv, tensor):
    x = tensor.unsqueeze(0).to(DEVICE)
    with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
        out = model.greedy_decode(x, conv.SOS, conv.EOS, conv.max_len + 2)
    return conv.decode(out[0].cpu().tolist())


def find_wins(m2d, m1d, conv, benchmarks, per_set=2):
    """Crops where 2D-RoPE is correct and 1D-sinusoidal is wrong."""
    wins = []
    for path, tag in benchmarks:
        got = 0
        for raw, tensor, gt in lmdb_samples(path, conv, limit=2000):
            p2 = predict(m2d, conv, tensor)
            p1 = predict(m1d, conv, tensor)
            gtn = normalize(gt)
            if normalize(p2) == gtn and normalize(p1) != gtn and len(gtn) >= 3:
                wins.append((raw, gt, p1, p2, tag))
                print(f"    [{tag}] GT={gt!r} 1D={p1!r} 2D={p2!r}")
                got += 1
                if got >= per_set:
                    break
        print(f"  {tag}: {got} win(s)")
    return wins


def render_qualitative(wins):
    # Two columns of samples (image + predictions) to keep the figure short
    # enough for a full-text-width float instead of a tall single stack.
    n = len(wins)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, 1.15 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx >= n:
            ax.axis("off"); continue
        raw, gt, p1, p2, tag = wins[idx]
        ax.imshow(raw)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(1.04, 0.78, f"GT: {gt}", transform=ax.transAxes,
                va="center", ha="left", family="monospace", fontsize=8)
        ax.text(1.04, 0.46, f"1D Sin.: {p1 or '∅'}", transform=ax.transAxes,
                va="center", ha="left", family="monospace", fontsize=8, color="#c0392b")
        ax.text(1.04, 0.14, f"2D-RoPE: {p2}", transform=ax.transAxes,
                va="center", ha="left", family="monospace", fontsize=8,
                color="#1f6f3f", weight="bold")
        ax.set_ylabel(tag, fontsize=8, rotation=0, ha="right", va="center")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.96, bottom=0.04,
                        wspace=1.6, hspace=0.5)
    out = os.path.join(OUT_DIR, "fig_qualitative.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def render_ablation():
    pe = json.load(open("results/abl_pe.json"))
    order = ["None", "1D Sinusoidal", "1D Learnable", "2D Sinusoidal",
             "2D Learnable", "1D RoPE", "2D-RoPE (Ours)"]
    names = [k for k in order if k in pe]
    avgs = [pe[k]["Avg"] for k in names]
    colors = ["#1f6f3f" if "Ours" in k else "#5b8fb9" if k.startswith("2D")
              else "#aaaaaa" for k in names]

    layers = json.load(open("results/abl_layers.json"))
    lk = sorted(layers.keys(), key=lambda s: int(s.split()[0]))
    lx = [int(k.split()[0]) for k in lk]
    ly = [layers[k]["Avg"] for k in lk]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    short = {"None": "None", "1D Sinusoidal": "1D Sin.", "1D Learnable": "1D Learn.",
             "2D Sinusoidal": "2D Sin.", "2D Learnable": "2D Learn.",
             "1D RoPE": "1D RoPE", "2D-RoPE (Ours)": "2D-RoPE\n(Ours)"}
    a.bar(range(len(names)), avgs, color=colors)
    a.set_xticks(range(len(names)))
    a.set_xticklabels([short[n] for n in names], fontsize=7, rotation=30, ha="right")
    a.set_ylim(min(avgs) - 0.3, max(avgs) + 0.25)
    a.set_ylabel("Avg. word acc. (%)", fontsize=9)
    a.set_title("(a) Positional encoding type", fontsize=9)
    a.grid(axis="y", ls=":", alpha=0.5)

    b.plot(lx, ly, "-o", color="#1f6f3f")
    best = lx[ly.index(max(ly))]
    b.axvline(6, ls="--", color="#888", lw=1)
    b.text(6, min(ly), " default", fontsize=7, color="#555", va="bottom")
    b.set_xticks(lx)
    b.set_xlabel("Encoder layers", fontsize=9)
    b.set_ylabel("Avg. word acc. (%)", fontsize=9)
    b.set_title("(b) Encoder depth", fontsize=9)
    b.grid(ls=":", alpha=0.5)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_ablation.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


@torch.no_grad()
def per_sample_correct(model, conv, dataset):
    """Boolean correctness per sample, aligned with what the loader returns."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4,
                        pin_memory=True)
    flags = []
    for images, _, tgt_out in loader:
        images = images.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
            pred = model.greedy_decode(images, conv.SOS, conv.EOS, conv.max_len + 2)
        for p, gt in zip(pred.cpu(), tgt_out):
            flags.append(normalize(conv.decode(p.tolist())) ==
                         normalize(conv.decode(gt.tolist())))
    return np.array(flags, dtype=bool)


def count_netwins(m2d, m1d, conv):
    """Per benchmark: #imgs 2D-RoPE gets right that 1D-sin gets wrong, and vice
    versa. Quantifies the qualitative figure so it isn't anecdotal."""
    root = "data_lmdb_release/evaluation"
    rows, agg = {}, [0, 0, 0]
    for dirname, name in EVAL_BENCHMARKS.items():
        path = os.path.join(root, dirname)
        if not os.path.isdir(path):
            continue
        ds = LmdbDataset(path, conv)
        c2 = per_sample_correct(m2d, conv, ds)
        c1 = per_sample_correct(m1d, conv, ds)
        win2 = int((c2 & ~c1).sum())   # 2D right, 1D wrong
        win1 = int((c1 & ~c2).sum())   # 1D right, 2D wrong
        rows[name] = {"n": len(c2), "2drope_only": win2, "1dsin_only": win1,
                      "net": win2 - win1}
        agg[0] += len(c2); agg[1] += win2; agg[2] += win1
        print(f"  {name:8s} n={len(c2):4d}  2D-only={win2:3d}  1D-only={win1:3d}"
              f"  net={win2-win1:+d}")
    rows["Total"] = {"n": agg[0], "2drope_only": agg[1], "1dsin_only": agg[2],
                     "net": agg[1] - agg[2]}
    print(f"  {'Total':8s} n={agg[0]:4d}  2D-only={agg[1]:3d}  1D-only={agg[2]:3d}"
          f"  net={agg[1]-agg[2]:+d}")
    json.dump(rows, open("results/netwins.json", "w"), indent=2)
    print("wrote results/netwins.json")
    return rows


# ---- attention extraction: monkeypatch SDPA to stash the softmax map ----
_CAPTURED = []


def _sdpa_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.size(-1))
    attn = scores.softmax(-1)
    _CAPTURED.append(attn.detach())
    return (attn @ v.float()).type_as(q)


@torch.no_grad()
def encoder_saliency(model, conv, tensor):
    """Mean last-layer encoder self-attention received per spatial cell -> (H,W)."""
    _CAPTURED.clear()
    orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa_capture
    try:
        model.encode(tensor.unsqueeze(0).to(DEVICE))
        H, W = model._mem_hw
    finally:
        F.scaled_dot_product_attention = orig
    attn = torch.stack(_CAPTURED).mean(0)[0]   # mean over all enc layers: (heads, N, N)
    sal = attn.mean(0).mean(0)                 # mean over heads and queries -> (N,)
    return sal.reshape(H, W).cpu().numpy(), (H, W)


def render_attention(model, conv, examples):
    """Encoder self-attention overlays from the full 2D-RoPE model (stride 4,
    8x32 grid). Shows that attention concentrates on the character strokes and
    follows the 2D text layout even under curvature/perspective."""
    # Lay the samples out in two rows: each sample is an (input, attention)
    # pair, so a row holds samples_per_row*2 axes. Wider and shorter than the
    # original single column of pairs.
    n = len(examples)
    nrows = 2
    per_row = (n + nrows - 1) // nrows          # samples per row
    fig, axes = plt.subplots(nrows, per_row * 2, figsize=(7.4, 1.3 * nrows))
    axes = np.array(axes).reshape(nrows, per_row * 2)
    for ax in axes.ravel():
        ax.axis("off")
    tf = build_transform(augment=False)
    for i, (raw, gt) in enumerate(examples):
        r, slot = divmod(i, per_row)
        sal, _ = encoder_saliency(model, conv, tf(raw))
        base = raw.resize((128, 32))
        for c, (title, s) in enumerate([("input", None), ("2D-RoPE attention", sal)]):
            ax = axes[r, slot * 2 + c]
            ax.axis("on")
            ax.imshow(base, extent=[0, 128, 32, 0])
            if s is not None:
                up = np.kron(s, np.ones((32 // s.shape[0], 128 // s.shape[1])))
                up = (up - up.min()) / (up.max() - up.min() + 1e-8)
                ax.imshow(up, cmap="jet", alpha=0.5, extent=[0, 128, 32, 0])
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"GT: {gt}", fontsize=8, family="monospace",
                              rotation=0, ha="right", va="center")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_attention.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _overlay(ax, raw, sal, vrange=None):
    """Draw the crop with (optional) jet attention overlay on a 128x32 canvas.
    vrange=(lo,hi) shares the colour scale across panels; else per-panel min-max."""
    base = raw.resize((128, 32))
    ax.imshow(base, extent=[0, 128, 32, 0])
    if sal is not None:
        up = np.kron(sal, np.ones((32 // sal.shape[0], 128 // sal.shape[1])))
        lo, hi = vrange if vrange else (up.min(), up.max())
        up = np.clip((up - lo) / (hi - lo + 1e-8), 0, 1)
        ax.imshow(up, cmap="jet", alpha=0.5, extent=[0, 128, 32, 0])
    ax.set_xticks([]); ax.set_yticks([])


def render_teaser(m1d, m2d, conv, raw, gt, p1, p2):
    """Page-1 teaser, grounded in real data: the same irregular crop fed to the
    PE-ablation pair (identical backbone/decoder/schedule, only the positional
    encoding differs). Left = 1D-sinusoidal (diffuse attention, wrong read);
    right = 2D-RoPE (focused attention, correct read); bottom = the rotary idea."""
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    tf = build_transform(augment=False)
    s1, _ = encoder_saliency(m1d, conv, tf(raw))
    s2, _ = encoder_saliency(m2d, conv, tf(raw))

    BLUE, TEAL = "#2b5fa8", "#1f8a82"
    RED, GREEN = "#c0392b", "#1f6f3f"
    fig = plt.figure(figsize=(3.5, 4.05))
    ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off")
    ov.set_xlim(0, 1); ov.set_ylim(0, 1)

    # ---- top: the real irregular crop ----
    axc = fig.add_axes([0.30, 0.845, 0.46, 0.11]); _overlay(axc, raw, None)
    for s in axc.spines.values():
        s.set_edgecolor("#888"); s.set_linewidth(0.8)
    ov.text(0.13, 0.90, "Irregular\nscene text", fontsize=7.5, style="italic",
            color="#555", ha="center", va="center")

    # feed arrows into the two panels
    for x0, x1 in [(0.42, 0.27), (0.62, 0.74)]:
        ov.add_patch(FancyArrowPatch((x0, 0.838), (x1, 0.775),
                     arrowstyle="-|>", mutation_scale=8, color="#999", lw=1.0))

    # ---- panel backgrounds ----
    ov.add_patch(FancyBboxPatch((0.05, 0.40), 0.42, 0.355,
                 boxstyle="round,pad=0.01", fc="#eef3fb", ec="#bcd", lw=0.8))
    ov.add_patch(FancyBboxPatch((0.53, 0.40), 0.42, 0.355,
                 boxstyle="round,pad=0.01", fc="#eefaf0", ec=GREEN, lw=1.1))
    ov.text(0.26, 0.725, "1D Positional Encoding\n(baseline)", fontsize=7.5,
            weight="bold", ha="center", va="center")
    ov.text(0.74, 0.725, "2D-RoPE (Ours)", fontsize=8, weight="bold",
            color=BLUE, ha="center", va="center")

    # ---- attention overlays ----
    vr = (min(s1.min(), s2.min()), max(s1.max(), s2.max()))
    axl = fig.add_axes([0.085, 0.55, 0.36, 0.105]); _overlay(axl, raw, s1, vr)
    axr = fig.add_axes([0.565, 0.55, 0.36, 0.105]); _overlay(axr, raw, s2, vr)
    ov.text(0.26, 0.515, "encoder self-attention", fontsize=6.5,
            color="#555", ha="center", va="center")
    ov.text(0.74, 0.515, "encoder self-attention", fontsize=6.5,
            color="#555", ha="center", va="center")
    ov.text(0.26, 0.445, f"{p1 or '∅'}  ✗", fontsize=8.5, family="monospace",
            color=RED, weight="bold", ha="center", va="center")
    ov.text(0.74, 0.445, f"{p2}  ✓", fontsize=8.5, family="monospace",
            color=GREEN, weight="bold", ha="center", va="center")

    # ---- key-idea band: rotary rotation ----
    ov.add_patch(FancyBboxPatch((0.05, 0.04), 0.90, 0.305,
                 boxstyle="round,pad=0.01", fc="#fffbe8", ec="#e6d8a0", lw=0.8))
    ov.text(0.085, 0.305, "Key idea", fontsize=7.5, weight="bold", va="center")
    # token at (h,w)
    cx, cy = 0.20, 0.165
    ov.add_patch(FancyBboxPatch((cx - 0.055, cy - 0.035), 0.11, 0.07,
                 boxstyle="round,pad=0.005", fc="#dce8fb", ec=BLUE, lw=0.8))
    ov.text(cx, cy, "$(h,w)$", fontsize=7.5, ha="center", va="center")
    # row rotation arc (above) + col rotation arc (right)
    ov.add_patch(FancyArrowPatch((cx - 0.075, cy + 0.045), (cx + 0.075, cy + 0.045),
                 connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                 mutation_scale=8, color=BLUE, lw=1.2))
    ov.text(cx, cy + 0.105, r"$\theta^h$ (row)", fontsize=6.5, color=BLUE, ha="center")
    ov.add_patch(FancyArrowPatch((cx + 0.085, cy + 0.04), (cx + 0.085, cy - 0.04),
                 connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                 mutation_scale=8, color=TEAL, lw=1.2))
    ov.text(cx + 0.155, cy + 0.065, r"$\theta^w$ (col)", fontsize=6.5,
            color=TEAL, ha="left", va="center")
    ov.text(0.665, 0.165,
            r"Rotate $q,k$ by row & column" "\n"
            r"position $\Rightarrow$ attention $\langle q,k\rangle$" "\n"
            r"depends only on offset $(\Delta h,\Delta w)$.",
            fontsize=6.6, ha="center", va="center")

    out_pdf = os.path.join(OUT_DIR, "fig_teaser.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_teaser.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}  (1D={p1!r} 2D={p2!r} gt={gt!r})")


def pick_irregular(conv, specs):
    """First distinct >=5-char word from each (benchmark_dir) for attention viz."""
    out, seen = [], set()
    for path in specs:
        for raw, _, gt in lmdb_samples(path, conv, limit=80):
            if gt not in seen and len(normalize(gt)) >= 5:
                out.append((raw, gt)); seen.add(gt)
                break
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    render_ablation()
    print("loading models...")
    m2d, conv = load_model("checkpoints/abl_pe_2d-rope-ours_best.pt")
    m1d, _ = load_model("checkpoints/abl_pe_1d-sinusoidal_best.pt")
    root = "data_lmdb_release/evaluation"
    benchmarks = [
        (f"{root}/CUTE80", "Curved"),
        (f"{root}/SVTP", "Perspective"),
        (f"{root}/IC15_2077", "Incidental"),
    ]
    print("scanning for 2D-RoPE wins...")
    wins = find_wins(m2d, m1d, conv, benchmarks, per_set=2)
    if wins:
        render_qualitative(wins)
        # page-1 teaser: prefer the COFFEE win, else the first perspective win
        teaser = next((w for w in wins if normalize(w[1]) == "coffee"),
                      next((w for w in wins if w[4] == "Perspective"), wins[0]))
        raw, gt, p1, p2, _ = teaser
        render_teaser(m1d, m2d, conv, raw, gt, p1, p2)
    else:
        print("no qualifying examples found")

    # attention maps from the strong full model (stride 4 -> crisp 8x32 grid)
    print("rendering attention maps (full 2D-RoPE model)...")
    m_full, conv_f = load_model("checkpoints/full_2drope_best.pt")
    ex = pick_irregular(conv_f, [f"{root}/CUTE80", f"{root}/CUTE80",
                                 f"{root}/SVTP", f"{root}/SVTP"])
    render_attention(m_full, conv_f, ex)

    print("counting net wins over all benchmarks...")
    count_netwins(m2d, m1d, conv)


if __name__ == "__main__":
    main()
