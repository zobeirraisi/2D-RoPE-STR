"""
Figure 1 of the IJDAR revision:
(a) how position enters the attention score for additive encodings, 1D RoPE, 2D-RoPE, and the
    key-only rotation in decoder cross-attention (Proposition 1);
(b) forest plot of every multi-seed controlled comparison in the paper: paired difference in
    average accuracy (mean, 95% CI, individual seeds), oriented so that positive favors 2D-RoPE
    or the adaptation.
Numbers are read from results/rev1/tables/{pe,pe_fs4,factorial,parseq}.json (written by
`make_tables.py`); never type numbers here.

  python make_overview_figure.py [--out ../manuscript/ijdar/revision1/fig_overview.pdf]
"""

import argparse
import json
import os
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

TEXTWIDTH_IN = 372 / 72.27            # sn-jnl \textwidth; included at \textwidth
INK, MUTED, LIGHT, GRID = "#222222", "#666666", "#b8b8b2", "#e3e3e0"
CELL, BLUE, BLUE_LT = "#f4f4f2", "#2a78d6", "#d6e5f8"
MINUS = "−"
GH, GW = 2, 5                         # illustrative grid; the real grids are 4x16 and 8x32
Q, K = (0, 1), (1, 3)                 # self-attention query and key cells: (dh, dw) = (1, 2)
K_CROSS = (0, 3)                      # cross-attention key (the decoder query sits above the grid)


def num(x, sign=False):
    s = f"{abs(x):.2f}"
    return (MINUS + s) if x < 0 else (("+" + s) if sign else s)


# ----------------------------------------------------------------------------- panel (a)
def grid(ax, x0, y0, c, label, q, k):
    """GH x GW token grid with its top-left corner at (x0, y0); c = cell size in inches."""
    for h in range(GH):
        for w in range(GW):
            is_q, is_k = (h, w) == q, (h, w) == k
            ax.add_patch(Rectangle((x0 + w * c, y0 - (h + 1) * c), c, c,
                                   fc=BLUE if is_q else (BLUE_LT if is_k else CELL),
                                   ec=BLUE if (is_q or is_k) else LIGHT,
                                   lw=1.1 if is_k else 0.5, zorder=2 if (is_q or is_k) else 1))
            ax.text(x0 + (w + 0.5) * c, y0 - (h + 0.5) * c, label(h, w), ha="center",
                    va="center", fontsize=6.5, color="white" if is_q else INK, zorder=3)


def tag(ax, x, y, s, va):
    ax.text(x, y, s, ha="center", va=va, fontsize=7, color=BLUE, weight="bold", style="italic")


def panel_a(ax, width, height):
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis("off")
    c, colw = 0.185, width / 4
    y_br, y_title = height - 0.07, height - 0.30
    y_grid = height - 0.53                                # top edge of the grids
    y_l1, y_l2 = y_grid - GH * c - 0.22, y_grid - GH * c - 0.38
    cols = [
        ("Additive (sin./learned)", lambda h, w: f"$p_{{{h * GW + w}}}$",
         r"$\mathbf{p}_m$ and $\mathbf{p}_n$", "absolute"),
        ("1D RoPE", lambda h, w: f"{h * GW + w}",
         r"$n-m = W'\Delta h + \Delta w$", "relative, one index"),
        ("2D-RoPE", lambda h, w: f"{h},{w}",
         r"$(\Delta h, \Delta w)$", "relative, per axis"),
        ("Key-only rotation", lambda h, w: f"{h},{w}",
         r"key position $(h, w)$", "absolute (Prop. 1)"),
    ]
    for i, (title, lab, l1, l2) in enumerate(cols):
        cx = (i + 0.5) * colw
        x0 = cx - GW * c / 2
        ax.text(cx, y_title, title, ha="center", va="center", fontsize=8, weight="bold", color=INK)
        if i < 3:
            grid(ax, x0, y_grid, c, lab, Q, K)
            tag(ax, x0 + (Q[1] + 0.5) * c, y_grid + 0.015, "q", "bottom")
            tag(ax, x0 + (K[1] + 0.5) * c, y_grid - GH * c - 0.015, "k", "top")
        else:
            grid(ax, x0, y_grid, c, lab, None, K_CROSS)
            # decoder query: a character slot with no 2D position, above the grid
            kx = x0 + (K_CROSS[1] + 0.5) * c
            bx, by = x0 + 1.0 * c, y_grid + 0.105
            ax.add_patch(FancyBboxPatch((bx - 0.15, by - 0.06), 0.30, 0.12,
                                        boxstyle="round,pad=0.005,rounding_size=0.025",
                                        fc=BLUE, ec=BLUE, lw=0.6, zorder=3))
            ax.text(bx, by - 0.005, r"$\mathbf{y}_t$", ha="center", va="center", fontsize=7,
                    color="white", zorder=4)
            ax.add_patch(FancyArrowPatch((bx + 0.15, by), (kx, y_grid - 0.01),
                                         connectionstyle="angle,angleA=0,angleB=90,rad=4",
                                         arrowstyle="-|>", mutation_scale=6, lw=0.9,
                                         color=BLUE, zorder=5, shrinkA=0, shrinkB=0))
            tag(ax, bx - 0.21, by, "q", "center")
            tag(ax, kx + 0.10, y_grid + 0.07, "k", "center")
        ax.text(cx, y_l1, l1, ha="center", va="center", fontsize=7.5, color=INK)
        ax.text(cx, y_l2, l2, ha="center", va="center", fontsize=7, color=MUTED)
    for (a, b), s in (((0.30, 3 * colw - 0.08), "Encoder self-attention"),
                      ((3 * colw + 0.08, width - 0.04), "Decoder cross-attention")):
        ax.plot([a, a, b, b], [y_br - 0.05, y_br, y_br, y_br - 0.05], color=LIGHT, lw=0.7)
        ax.text((a + b) / 2, y_br, s, ha="center", va="center", fontsize=7, color=MUTED,
                style="italic", bbox=dict(fc="white", ec="none", pad=1.2))
    ax.text(0.0, height, "(a)", fontsize=9, weight="bold", va="top", ha="left")


# ----------------------------------------------------------------------------- panel (b)
def comparisons(tables):
    load = lambda n: json.load(open(os.path.join(tables, f"{n}.json")))
    pe, fs4, fac, pq = load("pe"), load("pe_fs4"), load("factorial"), load("parseq")
    ref = pe["rows"]["2D-RoPE (Ours)"]["Avg"]["values"]
    enc = []
    for key, lab in {"1D Sinusoidal": "1D sinusoidal", "2D Sinusoidal": "2D sinusoidal",
                     "1D Learnable": "1D learnable", "None": "No encoding",
                     "2D Learnable": "2D learnable"}.items():
        e = pe["rows"][key]
        d = e["diff_to_2drope_avg"]                      # stored as other - 2D-RoPE: flip
        enc.append((lab, -d["mean"], (-d["ci95"][1], -d["ci95"][0]),
                    [r - o for r, o in zip(ref, e["Avg"]["values"])]))
    enc.sort(key=lambda r: r[1])
    e = pe["rows"]["1D RoPE"]; d = e["diff_to_2drope_avg"]
    rope4 = ("1D RoPE, 4×16 grid", -d["mean"], (-d["ci95"][1], -d["ci95"][0]),
             [r - o for r, o in zip(ref, e["Avg"]["values"])])
    d = fs4["diff_2d_minus_1d_avg"]
    rope8 = ("1D RoPE, 8×32 grid", d["mean"], tuple(d["ci95"]), d["values"])
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
        ("2D-RoPE vs. other encodings", [rope4, rope8] + enc),
        ("Adaptations (2×2 factorial)",
         [("Row-heavy split 48:16 vs. 32:32", es["estimate"], tuple(es["ci95"]), split),
          ("Rotated cross-attention keys vs. none", ec["estimate"], tuple(ec["ci95"]), cross)]),
        ("Transfer to PARSeq",
         [("2D-RoPE vs. learned embedding", d["mean"], tuple(d["ci95"]), d["values"])]),
    ]
    for _, rs in groups:                                 # consistency checks
        for lab, m, ci, s in rs:
            assert abs(mean(s) - m) < 1e-9, lab
            assert ci[0] < m < ci[1], lab
    return groups


def panel_b(fig, ax, groups, x_head, x_row):
    lab_tf = blended_transform_factory(fig.transFigure, ax.transData)
    y = 0.0
    for g, rows in groups:
        ax.text(x_head, y, g, transform=lab_tf, ha="left", va="center", fontsize=7.5,
                color=MUTED, style="italic")
        y -= 1.0
        for lab, m, ci, seeds in rows:
            ax.plot(ci, [y, y], color=INK, lw=1.1, solid_capstyle="butt", zorder=2)
            for x in ci:
                ax.plot([x, x], [y - 0.2, y + 0.2], color=INK, lw=0.9, zorder=2)
            ax.scatter(seeds, [y] * len(seeds), s=10, facecolors="white", edgecolors=MUTED,
                       lw=0.7, zorder=3)
            ax.scatter([m], [y], s=20, marker="s", color=BLUE, edgecolors="white", lw=0.4, zorder=4)
            ax.text(x_row, y, lab, transform=lab_tf, ha="left", va="center", fontsize=7.5, color=INK)
            ax.text(1.02, y, f"{num(m, sign=True)} [{num(ci[0])}, {num(ci[1])}]",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7, color=INK)
            y -= 1.0
        y -= 0.3
    ax.set_ylim(y + 0.6, 0.55)
    ax.set_yticks([])
    lo = min(min(ci[0], *s) for _, rs in groups for _, _, ci, s in rs)
    hi = max(max(ci[1], *s) for _, rs in groups for _, _, ci, s in rs)
    ax.set_xlim(lo - 0.1, hi + 0.1)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=2.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{MINUS}{abs(v):g}" if v < 0 else f"{v:g}"))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Paired difference in average accuracy (points)", fontsize=8, color=INK, labelpad=2)
    ax.annotate("", xy=(lo - 0.05, -0.245), xytext=(-0.05, -0.245), xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=5))
    ax.annotate("", xy=(hi + 0.05, -0.245), xytext=(0.05, -0.245), xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=5))
    ax.text(-0.08, -0.262, "favors the alternative", transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="right", va="top", fontsize=6.5, color=MUTED)
    ax.text(0.08, -0.262, "favors 2D-RoPE / adaptation", transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="left", va="top", fontsize=6.5, color=MUTED)
    ax.text(1.02, 0.0, "mean [95% CI]", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=7, color=MUTED)
    h = [Line2D([], [], marker="s", ls="", color=BLUE, ms=4, label="mean"),
         Line2D([], [], color=INK, lw=1.1, label="95% CI"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=3.3, label="each seed")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(x_head, 0.0), ncol=1, fontsize=6.5,
               frameon=False, handletextpad=0.4, labelspacing=0.25, handlelength=1.4, borderpad=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/rev1/tables")
    ap.add_argument("--out", default="../manuscript/ijdar/revision1/fig_overview.pdf")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "axes.linewidth": 0.6})
    groups = comparisons(args.tables)
    W, Ha, Hb = TEXTWIDTH_IN, 1.50, 2.45
    H = Ha + Hb
    fig = plt.figure(figsize=(W, H))
    panel_a(fig.add_axes([0.0, Hb / H, 1.0, Ha / H]), W, Ha)
    left, right, bottom, top = 2.10, 1.02, 0.50, 0.12     # inches
    ax_b = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (Hb - bottom - top) / H])
    panel_b(fig, ax_b, groups, x_head=0.02 / W, x_row=0.16 / W)
    fig.text(0.0, (Hb + 0.02) / H, "(b)", fontsize=9, weight="bold", va="top", ha="left")
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=250, bbox_inches="tight", pad_inches=0.02)
    for g, rs in groups:
        print(g)
        for lab, m, ci, s in rs:
            print(f"   {lab:40s} {num(m, True)} [{num(ci[0])}, {num(ci[1])}]  seeds {[round(x, 2) for x in s]}")


if __name__ == "__main__":
    main()
