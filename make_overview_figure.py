"""
Fig. 1 of the paper:
(a) how position enters the attention score for additive encodings, 1D RoPE, 2D-RoPE, and the
    key-only rotation in decoder cross-attention (Proposition 1);
(b) forest plot of every multi-seed controlled comparison in the paper: seed-paired difference in
    average accuracy (mean, 95% CI, individual seeds), oriented so that positive favors 2D-RoPE
    or the adaptation. The fixed-frequency row appears once pe_fixfreq.json exists.
Numbers are read from results/rev1/tables/{pe,pe_fs4,pe_fixfreq,factorial,parseq}.json (written by
`make_tables.py`); never type numbers here. The figure is drawn at the sn-jnl text width without a tight
bounding box, so the font sizes below are the printed sizes.

  python make_overview_figure.py [--out ../manuscript/ijdar/revision1/fig_overview.pdf]
"""

import argparse
import json
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

TEXTWIDTH_IN = 372 / 72.27
INK, MUTED, LIGHT, GRID = "#222222", "#666666", "#b8b8b2", "#e3e3e0"
CELL, BLUE, BLUE_LT = "#f4f4f2", "#2a78d6", "#d6e5f8"
MINUS = "−"
GH, GW = 2, 5
Q, K = (0, 1), (1, 3)
K_CROSS = (0, 3)
FS_CELL, FS_TXT, FS_HEAD, FS_PANEL = 7, 8, 8.5, 9


def num(x, sign=False):
    s = f"{abs(x):.2f}"
    return (MINUS + s) if x < 0 else (("+" + s) if sign else s)


# ----------------------------------------------------------------------------- panel (a)
def grid(ax, x0, y0, c, label, q, k):
    for h in range(GH):
        for w in range(GW):
            is_q, is_k = (h, w) == q, (h, w) == k
            ax.add_patch(Rectangle((x0 + w * c, y0 - (h + 1) * c), c, c,
                                   fc=BLUE if is_q else (BLUE_LT if is_k else CELL),
                                   ec=BLUE if (is_q or is_k) else LIGHT,
                                   lw=1.1 if is_k else 0.5, zorder=2 if (is_q or is_k) else 1))
            s = label(h, w, is_q, is_k)
            if s:
                ax.text(x0 + (w + 0.5) * c, y0 - (h + 0.5) * c, s, ha="center", va="center",
                        fontsize=FS_CELL, color="white" if is_q else INK, zorder=3)


def tag(ax, x, y, s, va):
    ax.text(x, y, s, ha="center", va=va, fontsize=FS_TXT, color=BLUE, weight="bold", style="italic")


def panel_a(ax, width, height):
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis("off")
    c, x_l, x_r = 0.19, 0.04, width - 0.02
    colw = (x_r - x_l) / 4
    y_br, y_title = height - 0.07, height - 0.29
    y_grid = height - 0.53
    y_l1, y_l2 = y_grid - GH * c - 0.20, y_grid - GH * c - 0.36
    cols = [
        ("Additive (1D/2D)",
         lambda h, w, iq, ik: r"$\mathbf{p}_m$" if iq else (r"$\mathbf{p}_n$" if ik else ""),
         r"$\mathbf{p}_m$ and $\mathbf{p}_n$", "absolute"),
        ("1D RoPE", lambda h, w, iq, ik: f"{h * GW + w}",
         r"$n-m = W'\Delta h + \Delta w$", "relative, one index"),
        ("2D-RoPE", lambda h, w, iq, ik: f"{h},{w}",
         r"$(\Delta h, \Delta w)$", "relative, per axis"),
        ("Key-only rotation", lambda h, w, iq, ik: f"{h},{w}",
         r"key position $(h, w)$", "absolute (Prop. 1)"),
    ]
    for i, (title, lab, l1, l2) in enumerate(cols):
        cx = x_l + (i + 0.5) * colw
        x0 = cx - GW * c / 2
        ax.text(cx, y_title, title, ha="center", va="center", fontsize=FS_HEAD, weight="bold", color=INK)
        if i < 3:
            grid(ax, x0, y_grid, c, lab, Q, K)
            tag(ax, x0 + (Q[1] + 0.5) * c, y_grid + 0.015, "q", "bottom")
            tag(ax, x0 + (K[1] + 0.5) * c, y_grid - GH * c - 0.015, "k", "top")
        else:
            grid(ax, x0, y_grid, c, lab, None, K_CROSS)
            kx = x0 + (K_CROSS[1] + 0.5) * c
            bx, by = x0 + 1.0 * c, y_grid + 0.105
            ax.add_patch(FancyBboxPatch((bx - 0.15, by - 0.06), 0.30, 0.12,
                                        boxstyle="round,pad=0.005,rounding_size=0.025",
                                        fc=BLUE, ec=BLUE, lw=0.6, zorder=3))
            ax.text(bx, by - 0.005, r"$\mathbf{y}_t$", ha="center", va="center", fontsize=FS_CELL,
                    color="white", zorder=4)
            ax.add_patch(FancyArrowPatch((bx + 0.15, by), (kx, y_grid - 0.01),
                                         connectionstyle="angle,angleA=0,angleB=90,rad=4",
                                         arrowstyle="-|>", mutation_scale=6, lw=0.9,
                                         color=BLUE, zorder=5, shrinkA=0, shrinkB=0))
            tag(ax, bx - 0.22, by, "q", "center")
            tag(ax, kx + 0.11, y_grid + 0.07, "k", "center")
        ax.text(cx, y_l1, l1, ha="center", va="center", fontsize=FS_TXT, color=INK)
        ax.text(cx, y_l2, l2, ha="center", va="center", fontsize=FS_CELL + 0.5, color=MUTED)
    for (a, b), s in (((x_l + 0.30, x_l + 3 * colw - 0.08), "Encoder self-attention"),
                      ((x_l + 3 * colw + 0.06, x_r), "Decoder cross-attention")):
        ax.plot([a, a, b, b], [y_br - 0.05, y_br, y_br, y_br - 0.05], color=LIGHT, lw=0.7)
        ax.text((a + b) / 2, y_br, s, ha="center", va="center", fontsize=FS_CELL, color=MUTED,
                style="italic", bbox=dict(fc="white", ec="none", pad=1.2))
    ax.text(0.0, height, "(a)", fontsize=FS_PANEL, weight="bold", va="top", ha="left")


# ----------------------------------------------------------------------------- panel (b)
def comparisons(tables):
    load = lambda n: json.load(open(os.path.join(tables, f"{n}.json")))
    pe, fs4, fac, pq = load("pe"), load("pe_fs4"), load("factorial"), load("parseq")
    ref = pe["rows"]["2D-RoPE (Ours)"]["Avg"]["values"]

    def vs_pe(key, lab):
        e = pe["rows"][key]
        d = e["diff_to_2drope_avg"]                      # stored as other - 2D-RoPE: flip
        return (lab, -d["mean"], (-d["ci95"][1], -d["ci95"][0]),
                [r - o for r, o in zip(ref, e["Avg"]["values"])])

    enc = [vs_pe(k, lab) for k, lab in {"1D Sinusoidal": "1D sinusoidal", "2D Sinusoidal": "2D sinusoidal",
                                        "1D Learnable": "1D learnable", "None": "No encoding",
                                        "2D Learnable": "2D learnable"}.items()]
    enc.sort(key=lambda r: r[1])
    d = fs4["diff_2d_minus_1d_avg"]
    rope = [vs_pe("1D RoPE", "4×16 grid"), ("8×32 grid", d["mean"], tuple(d["ci95"]), d["values"])]
    if os.path.exists(os.path.join(tables, "pe_fixfreq.json")):   # matched fixed-frequency control
        d = load("pe_fixfreq")["diff_fixed_minus_1drope_avg"]
        rope.insert(1, ("4×16 grid, fixed frequencies", d["mean"], tuple(d["ci95"]), d["values"]))
    cells = {k: v["Avg"]["values"] for k, v in fac["cells"].items()}
    per_seed = lambda a, b: [mean(x) - mean(y) for x, y in zip(zip(*a), zip(*b))]
    split = per_seed([cells["48:16 + no cross"], cells["48:16 + 2D cross"]],
                     [cells["32:32 + no cross"], cells["32:32 + 2D cross"]])
    cross = per_seed([cells["32:32 + 2D cross"], cells["48:16 + 2D cross"]],
                     [cells["32:32 + no cross"], cells["48:16 + no cross"]])
    ef = fac["effects"]["Avg"]
    es, ec = ef["split (48:16 - 32:32)"], ef["cross (2D keys - none)"]
    d = pq["1015_2077"]["2D-RoPE - learned (paired)"]["Avg"]
    groups = [
        ("2D-RoPE vs. 1D RoPE", rope),
        ("2D-RoPE vs. other encodings", enc),
        ("Adaptations (2×2 factorial)",
         [(r"Allocation $d_h{:}d_w$ 48:16 vs. 32:32", es["estimate"], tuple(es["ci95"]), split),
          ("Key-only rotation vs. none", ec["estimate"], tuple(ec["ci95"]), cross)]),
        ("PARSeq", [("2D-RoPE vs. learned embedding", d["mean"], tuple(d["ci95"]), d["values"])]),
    ]
    for _, rs in groups:
        for lab, m, ci, s in rs:
            assert abs(mean(s) - m) < 1e-9, lab
            assert ci[0] < m < ci[1], lab
    return groups


def panel_b(fig, ax, groups, x_head, x_row):
    lab_tf = blended_transform_factory(fig.transFigure, ax.transData)
    y = 0.0
    for g, rows in groups:
        ax.text(x_head, y, g, transform=lab_tf, ha="left", va="center", fontsize=FS_TXT,
                color=MUTED, style="italic")
        y -= 1.0
        for lab, m, ci, seeds in rows:
            ax.plot(ci, [y, y], color=INK, lw=1.1, solid_capstyle="butt", zorder=2)
            for x in ci:
                ax.plot([x, x], [y - 0.2, y + 0.2], color=INK, lw=0.9, zorder=2)
            ax.scatter(seeds, [y] * len(seeds), s=11, facecolors="white", edgecolors=MUTED,
                       lw=0.7, zorder=3)
            ax.scatter([m], [y], s=22, marker="s", color=BLUE, edgecolors="white", lw=0.4, zorder=4)
            ax.text(x_row, y, lab, transform=lab_tf, ha="left", va="center", fontsize=FS_TXT, color=INK)
            y -= 1.0
        y -= 0.25
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_yticks([])
    lo = min(min(ci[0], *s) for _, rs in groups for _, _, ci, s in rs)
    hi = max(max(ci[1], *s) for _, rs in groups for _, _, ci, s in rs)
    ax.set_xlim(lo - 0.1, hi + 0.1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="x", labelsize=FS_CELL + 0.5, colors=MUTED, length=2.5, pad=1.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{MINUS}{abs(v):g}" if v < 0 else f"{v:g}"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Seed-paired difference in average accuracy (points)", fontsize=FS_TXT, color=INK, labelpad=2)
    # direction cue on the first group's header line, either side of zero
    tf = blended_transform_factory(ax.transData, ax.transData)
    ax.text(-0.06, 0.0, "← alternative better", transform=tf, ha="right", va="center",
            fontsize=FS_CELL, color=MUTED)
    ax.text(0.06, 0.0, "2D-RoPE / adaptation better →", transform=tf, ha="left", va="center",
            fontsize=FS_CELL, color=MUTED)


def legend_row(fig, x_in, y_in, W, H):
    """One-line legend, positions in inches: blue square = mean, bar = 95% CI, circle = seed."""
    w, h = 1.85, 0.16
    ax = fig.add_axes([x_in / W, (y_in - h / 2) / H, w / W, h / H]); ax.axis("off")
    ax.set_xlim(0, w); ax.set_ylim(-1, 1)
    ax.scatter([0.04], [0], s=22, marker="s", color=BLUE, clip_on=False)
    ax.text(0.10, 0, "mean", fontsize=FS_CELL, va="center", color=INK)
    ax.plot([0.52, 0.72], [0, 0], color=INK, lw=1.1)
    for xx in (0.52, 0.72):
        ax.plot([xx, xx], [-0.4, 0.4], color=INK, lw=0.9)
    ax.text(0.78, 0, "95% CI", fontsize=FS_CELL, va="center", color=INK)
    ax.scatter([1.30], [0], s=11, facecolors="white", edgecolors=MUTED, lw=0.7, clip_on=False)
    ax.text(1.36, 0, "each seed", fontsize=FS_CELL, va="center", color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/rev1/tables")
    ap.add_argument("--out", default="../manuscript/ijdar/revision1/fig_overview.pdf")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": FS_TXT, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "axes.linewidth": 0.6})
    groups = comparisons(args.tables)
    n_rows = sum(len(rs) for _, rs in groups)
    W, Ha, Hb = TEXTWIDTH_IN, 1.45, 2.35 + 0.126 * (n_rows - 10)   # keep the row pitch fixed
    H = Ha + Hb
    fig = plt.figure(figsize=(W, H))
    panel_a(fig.add_axes([0.0, Hb / H, 1.0, Ha / H]), W, Ha)
    left, right, bottom, top = 2.05, 0.12, 0.36, 0.10     # inches
    ax_b = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (Hb - bottom - top) / H])
    panel_b(fig, ax_b, groups, x_head=0.02 / W, x_row=0.16 / W)
    legend_row(fig, 0.04, 0.12, W, H)
    fig.text(0.0, (Hb + 0.02) / H, "(b)", fontsize=FS_PANEL, weight="bold", va="top", ha="left")
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=250)
    for g, rs in groups:
        print(g)
        for lab, m, ci, s in rs:
            print(f"   {lab:40s} {num(m, True)} [{num(ci[0])}, {num(ci[1])}]  seeds {[round(x, 2) for x in s]}")


if __name__ == "__main__":
    main()
