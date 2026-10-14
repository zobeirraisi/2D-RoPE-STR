"""
E4 figure for the IJDAR revision: accuracy under controlled rotation / perspective (top) and the
seed-paired difference in accuracy drop relative to 2D-RoPE (bottom). The top panels also show the
full-schedule checkpoints (Amendment 4; one run each, dashed, hollow markers). Reads only
results/rev1/tables/geom.json (written by `make_tables.py geom`); never type numbers here.

  python make_geom_figure.py [--out ../manuscript/ijdar/revision1/fig_geom.pdf]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed series order and colors (validated categorical palette; markers give a second encoding).
SERIES = [
    ("2D-RoPE (Ours)", "2D-RoPE", "#2a78d6", "o"),
    ("1D RoPE", "1D RoPE", "#eb6834", "s"),
    ("1D Sinusoidal", "1D sinusoidal", "#1baf7a", "^"),
    ("None", "No PE", "#eda100", "D"),
]
LEVELS = {"rotation": [0, 10, 20, 30, 45, 60], "perspective": [0, 0.1, 0.2, 0.3, 0.4]}
XLABEL = {"rotation": r"rotation angle $\alpha$ (degrees)", "perspective": r"perspective strength $s$"}
FULL = {"2D-RoPE (Ours)": "2D-RoPE, seed 0", "1D RoPE": "1D RoPE, seed 0", "1D Sinusoidal": "1D sinusoidal, seed 0"}
TRAIN_MAX = {"rotation": 15, "perspective": 0.3}  # edge of the training augmentation
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="results/rev1/tables/geom.json")
    ap.add_argument("--out", default="../manuscript/ijdar/revision1/fig_geom.pdf")
    args = ap.parse_args()
    st = json.load(open(args.geom))
    rows, paired = st["rows"], st["paired_drop_vs_2drope"]

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex="col",
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for c, kind in enumerate(["rotation", "perspective"]):
        lv = LEVELS[kind]
        ax = axes[0, c]
        for key, name, col, mk in SERIES:
            e = rows[key]
            m = np.array([e["clean"]["mean"]] + [e[f"{kind}:{x}"]["mean"] for x in lv[1:]])
            sd = np.array([e["clean"]["sd"]] + [e[f"{kind}:{x}"]["sd"] for x in lv[1:]])
            ax.fill_between(lv, m - sd, m + sd, color=col, alpha=0.18, lw=0)
            ax.plot(lv, m, color=col, lw=1.5, marker=mk, ms=4, label=name)
            f = st.get("full_schedule", {}).get("rows", {}).get(FULL.get(key))
            if f:
                mf = [f["clean"]] + [f[f"{kind}:{x}"]["acc"] for x in lv[1:]]
                ax.plot(lv, mf, color=col, lw=1.0, ls="--", marker=mk, ms=4, mfc="white")
        ax.axvline(TRAIN_MAX[kind], color=MUTED, lw=0.8, ls=":")
        ax.set_ylabel("Avg. word accuracy (%)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(kind.capitalize(), color=INK, fontsize=9)

        ax = axes[1, c]
        ax.axhline(0, color=MUTED, lw=0.8)
        for i, (key, name, col, mk) in enumerate(SERIES[1:]):
            d = paired[key]
            x = np.array(lv[1:], dtype=float)
            off = (i - 1) * (1.2 if kind == "rotation" else 0.008)  # dodge overlapping CIs
            m = np.array([d[f"{kind}:{v}"]["mean"] for v in lv[1:]])
            lo = np.array([d[f"{kind}:{v}"]["ci95"][0] for v in lv[1:]])
            hi = np.array([d[f"{kind}:{v}"]["ci95"][1] for v in lv[1:]])
            ax.errorbar(x + off, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=4, lw=0,
                        elinewidth=1.0, capsize=2)
        ax.axvline(TRAIN_MAX[kind], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel(XLABEL[kind])
        ax.set_ylabel("extra drop vs.\n2D-RoPE (points)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xticks(lv)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    h, lab = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=MUTED, lw=1.5), plt.Line2D([], [], color=MUTED, lw=1.0, ls="--", marker="o", mfc="white", ms=4)]
    lab += ["reduced schedule, 3 seeds", "full schedule, 1 run"]
    axes[0, 0].legend(h, lab, frameon=False, loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
