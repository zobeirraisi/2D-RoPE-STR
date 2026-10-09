"""
Generate the multi-seed tables and statistics of the IJDAR revision from the result JSONs.
Never type these numbers by hand.

  python make_tables.py factorial   # E2: Table 5 rows + main effects / interaction
  python make_tables.py pe          # E1: multi-seed column of Table 3 + paired differences to 2D-RoPE

Outputs go to results/rev1/tables/ (<study>.json with all statistics, <study>_rows.tex).

Analysis of the 2x2 factorial (fixed before the E2 results were seen):
  outcomes   Avg (unweighted mean of six), Regular = pooled accuracy over the 4,662 images of
             IIIT5K+SVT+IC13, Irregular = pooled accuracy over the 3,010 images of
             IC15+CUTE80+SVTP, and each benchmark
  model      y = cell + seed + error (randomised block design: a given seed fixes data order
             and initialisation in all four cells), residual df = 3 x 2 = 6
  effects    split = mean(48:16) - mean(32:32); cross = mean(2D-RoPE keys) - mean(none);
             interaction = (cross effect at 48:16) - (cross effect at 32:32)
             each with SE from the residual mean square, 95% CI (t, df=6) and two-sided p
  primary    Avg; Regular and Irregular are secondary; per-benchmark effects are descriptive
             (nine outcome x effect tests: expect ~1 spurious p<0.05 by chance)
"""

import argparse
import json
import math
import os

import numpy as np

BENCH = ["IIIT5K", "SVT", "IC13", "IC15", "CUTE80", "SVTP"]
N_IMG = {"IIIT5K": 3000, "SVT": 647, "IC13": 1015, "IC15": 2077, "CUTE80": 288, "SVTP": 645}
REGULAR, IRREGULAR = ["IIIT5K", "SVT", "IC13"], ["IC15", "CUTE80", "SVTP"]
OUTCOMES = ["Avg", "Regular", "Irregular"] + BENCH

FACTORIAL_CELLS = [  # (label in the JSON, split, cross keys)
    ("32:32 + no cross", "$32{:}32$", "none"),
    ("48:16 + no cross", "$48{:}16$", "none"),
    ("32:32 + 2D cross", "$32{:}32$", "2D-RoPE"),
    ("48:16 + 2D cross", "$48{:}16$", "2D-RoPE"),
]
PE_ROWS = ["None", "1D Sinusoidal", "1D Learnable", "2D Sinusoidal", "2D Learnable", "1D RoPE", "2D-RoPE (Ours)"]


# ---------------------------------------------------------------- Student t (no scipy)
def t_cdf(t, df):
    """CDF of Student's t by integrating the density (accurate to ~1e-7 for |t| < 50)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    x = np.linspace(0.0, abs(t), 20001)
    area = np.trapezoid(c * (1 + x ** 2 / df) ** (-(df + 1) / 2), x)
    return 0.5 + math.copysign(area, t)


def t_two_sided_p(t, df):
    return 2 * (1 - t_cdf(abs(t), df))


def t_crit(df, q=0.975):
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- helpers
def outcomes(r):
    out = {b: r[b] for b in BENCH}
    out["Avg"] = sum(r[b] for b in BENCH) / len(BENCH)
    for name, group in (("Regular", REGULAR), ("Irregular", IRREGULAR)):
        out[name] = sum(r[b] * N_IMG[b] for b in group) / sum(N_IMG[b] for b in group)
    return out


def load_seeds(res_dir, stem, seeds):
    found = {}
    for s in seeds:
        p = os.path.join(res_dir, f"{stem}_s{s}.json")
        if os.path.exists(p):
            with open(p) as f:
                found[s] = json.load(f)
    return found


def fmt(m, sd, dec):
    return f"{m:.{dec}f}" if sd is None else f"{m:.{dec}f}{{\\scriptsize$\\pm${sd:.{dec}f}}}"


# ---------------------------------------------------------------- factorial
def factorial(res_dir, out_dir, seeds):
    runs = load_seeds(res_dir, "abl_factorial", seeds)
    seeds = [s for s in seeds if s in runs and all(c[0] in runs[s] for c in FACTORIAL_CELLS)]
    if not seeds:
        raise SystemExit("no complete factorial seed found")
    print(f"factorial: complete seeds {seeds}")
    # y[outcome] has shape (split 2, cross 2, seeds)
    y = {o: np.zeros((2, 2, len(seeds))) for o in OUTCOMES}
    for k, s in enumerate(seeds):
        for label, split, cross in FACTORIAL_CELLS:
            i, j = int(split != "$32{:}32$"), int(cross != "none")
            for o, v in outcomes(runs[s][label]).items():
                y[o][i, j, k] = v

    stats = {"seeds": seeds, "cells": {}, "effects": {}}
    rows = []
    for label, split, cross in FACTORIAL_CELLS:
        i, j = int(split != "$32{:}32$"), int(cross != "none")
        cell = {o: {"mean": float(y[o][i, j].mean()),
                    "sd": float(y[o][i, j].std(ddof=1)) if len(seeds) > 1 else None,
                    "values": y[o][i, j].tolist()} for o in OUTCOMES}
        stats["cells"][label] = cell
        cells = [fmt(cell[b]["mean"], cell[b]["sd"], 1) for b in BENCH] + [fmt(cell["Avg"]["mean"], cell["Avg"]["sd"], 2)]
        rows.append(f"{split} & {cross} & " + " & ".join(cells) + r" \\")

    n = len(seeds)
    if n >= 2:
        df = 3 * (n - 1)
        tc = t_crit(df)
        for o in OUTCOMES:
            a = y[o]
            resid = a - a.mean(2, keepdims=True) - a.mean((0, 1), keepdims=True) + a.mean()
            mse = (resid ** 2).sum() / df
            ests = {
                "split (48:16 - 32:32)": (a[1].mean() - a[0].mean(), math.sqrt(mse * (2 / (2 * n)))),
                "cross (2D keys - none)": (a[:, 1].mean() - a[:, 0].mean(), math.sqrt(mse * (2 / (2 * n)))),
                "interaction (cross effect at 48:16 - at 32:32)":
                    ((a[1, 1] - a[1, 0] - a[0, 1] + a[0, 0]).mean(), math.sqrt(mse * 4 / n)),
            }
            stats["effects"][o] = {}
            for name, (est, se) in ests.items():
                t = est / se if se > 0 else float("inf")
                stats["effects"][o][name] = {"estimate": float(est), "se": se, "df": df, "t": t,
                                             "ci95": [est - tc * se, est + tc * se],
                                             "p": t_two_sided_p(t, df) if math.isfinite(t) else 0.0}
    _write(out_dir, "factorial", stats, rows)
    print("\n".join(rows))
    for o in ("Avg", "Regular", "Irregular"):
        for name, e in stats["effects"].get(o, {}).items():
            print(f"  {o:9s} {name:48s} {e['estimate']:+6.2f}  95% CI [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]  p={e['p']:.3f}")


# ---------------------------------------------------------------- PE ablation
def pe(res_dir, out_dir, seeds):
    runs = load_seeds(res_dir, "abl_pe", seeds)
    print(f"pe: seeds with results {sorted(runs)}")
    stats, rows = {"rows": {}}, []
    ref = "2D-RoPE (Ours)"
    for label in PE_ROWS:
        ss = [s for s in sorted(runs) if label in runs[s]]
        vals = {o: np.array([outcomes(runs[s][label])[o] for s in ss]) for o in OUTCOMES} if ss else {}
        entry = {"seeds": ss}
        for o, v in vals.items():
            entry[o] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "values": v.tolist()}
        # paired difference to 2D-RoPE over the seeds both have
        common = [s for s in ss if ref in runs[s]]
        if label != ref and len(common) >= 2:
            d = np.array([outcomes(runs[s][label])["Avg"] - outcomes(runs[s][ref])["Avg"] for s in common])
            se = d.std(ddof=1) / math.sqrt(len(d))
            df = len(d) - 1
            t = d.mean() / se if se > 0 else float("inf")
            entry["diff_to_2drope_avg"] = {"seeds": common, "mean": float(d.mean()), "se": float(se), "df": df,
                                           "ci95": [d.mean() - t_crit(df) * se, d.mean() + t_crit(df) * se],
                                           "p": t_two_sided_p(t, df) if math.isfinite(t) else 0.0}
        stats["rows"][label] = entry
        rows.append(f"{label} & " + (fmt(entry["Avg"]["mean"], entry["Avg"]["sd"], 2) if ss else r"\pending{--}")
                    + f" & {len(ss)}" + r" \\")
    _write(out_dir, "pe", stats, rows)
    for label, e in stats["rows"].items():
        if "Avg" in e:
            dd = e.get("diff_to_2drope_avg")
            extra = f"   vs 2D-RoPE {dd['mean']:+.2f} [{dd['ci95'][0]:+.2f}, {dd['ci95'][1]:+.2f}] p={dd['p']:.3f}" if dd else ""
            sd = f"{e['Avg']['sd']:.2f}" if e['Avg']['sd'] is not None else "-"
            print(f"  {label:16s} n={len(e['seeds'])}  Avg {e['Avg']['mean']:.2f} +- {sd}{extra}")


def _write(out_dir, name, stats, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    with open(os.path.join(out_dir, f"{name}_rows.tex"), "w") as f:
        f.write("% generated by code/make_tables.py -- do not edit\n" + "\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["factorial", "pe"])
    ap.add_argument("--results", default="results/rev1")
    ap.add_argument("--out_dir", default="results/rev1/tables")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    {"factorial": factorial, "pe": pe}[args.study](args.results, args.out_dir, args.seeds)


if __name__ == "__main__":
    main()
