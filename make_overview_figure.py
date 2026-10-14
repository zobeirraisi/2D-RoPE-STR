"""
Figure 1 of the IJDAR revision: (a) how each family of positional encodings injects position
on the text feature grid, and (b) the headline result of the controlled comparison
(seven encodings, three seeds). Panel (b) reads only results/rev1/tables/pe.json
(written by `make_tables.py pe`); never type numbers here.

  python make_overview_figure.py [--out ../manuscript/ijdar/revision1/fig_overview.pdf]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TEXTWIDTH_IN = 372 / 72.27  # sn-jnl \textwidth; the figure is included at \textwidth
INK, MUTED, GRID, CELL = "#222222", "#666666", "#e3e3e0", "#f4f4f2"
ROTARY, ADDITIVE, NONE = "#2a78d6", "#eb6834", "#8a8a85"   # validated categorical slots + neutral
GROUP = {"None": NONE, "1D Sinusoidal": ADDITIVE, "1D Learnable": ADDITIVE, "2D Sinusoidal": ADDITIVE,
         "2D Learnable": ADDITIVE, "1D RoPE": ROTARY, "2D-RoPE (Ours)": ROTARY}
LABEL = {"2D-RoPE (Ours)": "2D-RoPE"}
GH, GW = 2, 5          # illustrative grid (the real grids are 4x16 and 8x32)


def draw_grid(ax, x0, y0, cell, labeler, hl=(), color=ROTARY):
    for h in range(GH):
        for w in range(GW):
            on = (h, w) in hl
            ax.add_patch(Rectangle((x0 + w * cell, y0 - (h + 1) * cell), cell, cell,
                                   fc=("#dbe8fa" if on else CELL), ec=(color if on else "#b8b8b2"),
                                   lw=(1.0 if on else 0.6)))
            ax.text(x0 + (w + 0.5) * cell, y0 - (h + 0.5) * cell, labeler(h, w),
                    ha="center", va="center", fontsize=7.5, color=INK)


def panel_a(ax):
    ax.set_xlim(0, 3); ax.set_ylim(0, 1); ax.axis("off")
    cell = 0.135
    cols = [
        ("Additive (sin. / learned)", lambda h, w: f"+$p_{{{h * GW + w}}}$", (),
         r"$\mathbf{x}_n+\mathbf{p}_n$ (absolute)", None, None),
        ("1D RoPE", lambda h, w: f"{h * GW + w}", ((0, 1), (1, 1)),
         r"score depends on $n-m$", None, r"vertical pair: $\Delta n = W' = 5$"),
        ("2D-RoPE", lambda h, w: f"{h},{w}", ((0, 1), (1, 1)),
         r"score depends on $(\Delta h, \Delta w)$", None,
         r"vertical pair: $(\Delta h, \Delta w) = (1, 0)$"),
    ]
    for i, (title, lab, hl, line1, line2, line3) in enumerate(cols):
        cx = i + 0.5
        x0 = cx - GW * cell / 2
        ax.text(cx, 0.99, title, ha="center", va="top", fontsize=8.5, weight="bold",
                color=(ROTARY if i else ADDITIVE))
        draw_grid(ax, x0, 0.80, cell, lab, hl)
        ax.text(cx, 0.30, line1, ha="center", va="center", fontsize=8, color=INK)
        if line2:
            ax.text(cx, 0.28, line2, ha="center", va="center", fontsize=7.5, color=MUTED)
        if line3:
            ax.text(cx, 0.10, line3, ha="center", va="center", fontsize=7.5, color=ROTARY)
    ax.text(0.0, 1.07, "(a)", transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")


def panel_b(ax, pe):
    rows = sorted(pe["rows"].items(), key=lambda kv: kv[1]["Avg"]["mean"])
    for y, (name, e) in enumerate(rows):
        c = GROUP[name]
        m, sd, vals = e["Avg"]["mean"], e["Avg"]["sd"], e["Avg"]["values"]
        ax.plot([m - sd, m + sd], [y, y], color=c, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter(vals, [y] * len(vals), s=16, facecolors="white", edgecolors=c, lw=0.9, zorder=3)
        ax.scatter([m], [y], s=30, color=c, edgecolors="white", lw=0.8, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(n, n) for n, _ in rows], fontsize=8)
    for t, (n, _) in zip(ax.get_yticklabels(), rows):
        t.set_color(INK)
        if n in ("1D RoPE", "2D-RoPE (Ours)"):
            t.set_weight("bold")
    means = [e["Avg"]["mean"] for _, e in rows]
    allv = [v for _, e in rows for v in e["Avg"]["values"]]
    ax.set_xlim(min(allv) - 0.15, max(allv) + 0.15)
    ax.set_xlabel("Avg. word accuracy (%)", fontsize=8, color=INK)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    # numbers for the caption (printed, not drawn)
    all_ci_zero = all(e["diff_to_2drope_avg"]["ci95"][0] <= 0 <= e["diff_to_2drope_avg"]["ci95"][1]
                      for n, e in pe["rows"].items() if n != "2D-RoPE (Ours)")
    print(f"caption: means span {max(means) - min(means):.2f} points; all paired CIs contain 0: {all_ci_zero}")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=ROTARY, ms=4.5, label="rotary"),
         Line2D([], [], marker="o", ls="", color=ADDITIVE, ms=4.5, label="additive"),
         Line2D([], [], marker="o", ls="", color=NONE, ms=4.5, label="none"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=4, label="single seed"),
         Line2D([], [], color=MUTED, lw=1.6, label=r"mean $\pm$ sd")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.36, 1.0), fontsize=7.5, frameon=False,
              ncol=5, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
    ax.text(-0.33, 1.05, "(b)", transform=ax.transAxes, fontsize=9, weight="bold", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", default="results/rev1/tables/pe.json")
    ap.add_argument("--out", default="../manuscript/ijdar/revision1/fig_overview.pdf")
    args = ap.parse_args()
    pe = json.load(open(args.pe))
    plt.rcParams.update({"font.size": 8, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.7))
    ax_a = fig.add_axes([0.0, 0.66, 1.0, 0.33])
    ax_b = fig.add_axes([0.25, 0.09, 0.72, 0.44])
    panel_a(ax_a)
    panel_b(ax_b, pe)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
