"""
Generate the multi-seed tables and statistics of the IJDAR revision from the result JSONs.
Never type these numbers by hand.

  python make_tables.py factorial   # E2: Table 5 rows + main effects / interaction
  python make_tables.py pe          # E1: multi-seed column of Table 3 + paired differences to 2D-RoPE
  python make_tables.py geom        # E4: rotation / perspective sweep, mean +- sd over training seeds
                                    #     + seed-paired differences in the drop from clean accuracy
  python make_tables.py full        # E3: full-schedule rows of Table 1 (one run each, 1 decimal)
  python make_tables.py xattn       # E2 diagnostic: per-layer cross-attention profile, 2D keys vs none
  python make_tables.py pe_fs4      # E7: 1D vs 2D RoPE at stride 4, paired by seed (Amendment 3 rule)
  python make_tables.py parseq      # E5: PARSeq learned PE vs 2D-RoPE, mean +- sd over seeds
                                    #     + seed-paired difference (reads ../third_party/parseq/results)

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


# ---------------------------------------------------------------- E7 1D vs 2D RoPE at stride 4
def pe_fs4(res_dir, out_dir, seeds):
    """E7 (Amendment 3 rule, fixed before any E7 result): 2D-RoPE - 1D RoPE at feature stride 4,
    paired by seed. 'Helps' only if the mean paired Avg difference >= 0.5 AND 2D > 1D in every seed."""
    runs = load_seeds(res_dir, "abl_pe_fs4", seeds)
    common = sorted(s for s in runs if {"1D RoPE", "2D-RoPE (Ours)"} <= set(runs[s]))
    if len(common) < 2:
        raise SystemExit(f"need >= 2 seeds with both arms in {res_dir}, found {common}")
    stats, rows = {"seeds": common, "rows": {}}, []
    for label in ["1D RoPE", "2D-RoPE (Ours)"]:
        v = {o: np.array([outcomes(runs[s][label])[o] for s in common]) for o in OUTCOMES}
        stats["rows"][label] = {o: {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "values": x.tolist()}
                                for o, x in v.items()}
        rows.append(f"{label} & " + " & ".join(fmt(v[b].mean(), v[b].std(ddof=1), 1) for b in BENCH)
                    + " & " + fmt(v["Avg"].mean(), v["Avg"].std(ddof=1), 2) + f" & {len(common)} \\\\")
    d = np.array([outcomes(runs[s]["2D-RoPE (Ours)"])["Avg"] - outcomes(runs[s]["1D RoPE"])["Avg"] for s in common])
    diff = _paired(d)
    helps = bool(diff["mean"] >= 0.5 and (d > 0).all())
    stats["diff_2d_minus_1d_avg"] = diff
    stats["decision"] = "2D helps at stride 4" if helps else "no difference at stride 4"
    _write(out_dir, "pe_fs4", stats, rows)
    print(f"pe_fs4: seeds {common}")
    print(f"  1D RoPE {stats['rows']['1D RoPE']['Avg']['mean']:.2f}  2D-RoPE {stats['rows']['2D-RoPE (Ours)']['Avg']['mean']:.2f}"
          f"  diff {diff['mean']:+.2f} (per seed {', '.join(f'{x:+.2f}' for x in d)})"
          f"  95% CI [{diff['ci95'][0]:+.2f}, {diff['ci95'][1]:+.2f}]  p={diff['p']:.3f}")
    print(f"  decision: {stats['decision']}")


# ---------------------------------------------------------------- E4 geometric robustness
# (label in the table, JSON file tag). Factorial cells are evaluated in the same sweep.
GEOM_ROWS = [
    ("None", "abl_pe_none"), ("1D Sinusoidal", "abl_pe_1d-sinusoidal"), ("1D Learnable", "abl_pe_1d-learnable"),
    ("2D Sinusoidal", "abl_pe_2d-sinusoidal"), ("2D Learnable", "abl_pe_2d-learnable"),
    ("1D RoPE", "abl_pe_1d-rope"), ("2D-RoPE (Ours)", "abl_pe_2d-rope-ours"),
    ("$32{:}32$ + no cross", "abl_factorial_32-32-no-cross"), ("$48{:}16$ + no cross", "abl_factorial_48-16-no-cross"),
    ("$32{:}32$ + 2D cross", "abl_factorial_32-32-2d-cross"), ("$48{:}16$ + 2D cross", "abl_factorial_48-16-2d-cross"),
]
GEOM_LEVELS = {"rotation": [10, 20, 30, 45, 60], "perspective": [0.1, 0.2, 0.3, 0.4]}


def geom(res_dir, out_dir, seeds):
    geo_dir = os.path.join(res_dir, "geom")
    stats, rows = {"clean": {}, "rows": {}}, {"rotation": [], "perspective": []}
    for label, tag in GEOM_ROWS:
        runs = load_seeds(geo_dir, tag, seeds)
        if not runs:
            continue
        ss = sorted(runs)
        entry = {"seeds": ss}
        for kind, levels in GEOM_LEVELS.items():
            # clean accuracy is stored once per run, under rotation level 0
            clean = np.array([outcomes(runs[s]["results"]["rotation"]["0"]["-"])["Avg"] for s in ss])
            entry.setdefault("clean", {"mean": float(clean.mean()), "sd": float(clean.std(ddof=1)) if len(ss) > 1 else None,
                                       "values": clean.tolist()})
            for lv in levels:
                key = str(lv)
                # mean over the test-time seeds of each run, then mean +- sd over training seeds
                v = np.array([np.mean([outcomes(runs[s]["results"][kind][key][str(t)])["Avg"]
                                       for t in runs[s]["tseeds"]]) for s in ss])
                entry[f"{kind}:{lv}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(ss) > 1 else None,
                                         "values": v.tolist(), "drop_from_clean": float((clean - v).mean())}
        stats["rows"][label] = entry
        for kind, levels in GEOM_LEVELS.items():
            cells = [fmt(entry["clean"]["mean"], entry["clean"]["sd"], 2)]
            cells += [fmt(entry[f"{kind}:{lv}"]["mean"], entry[f"{kind}:{lv}"]["sd"], 2) for lv in levels]
            rows[kind].append(f"{label} & " + " & ".join(cells) + r" \\")
    if not stats["rows"]:
        raise SystemExit(f"no geom results in {geo_dir}")
    # Does 2D-RoPE degrade more slowly? Paired by training seed: d = drop(other) - drop(2D-RoPE),
    # positive = 2D-RoPE loses less accuracy than the other encoding at that strength.
    stats["paired_drop_vs_2drope"] = {}
    ref = stats["rows"].get("2D-RoPE (Ours)")
    if ref:
        for other in ["None", "1D Sinusoidal", "1D RoPE", "2D Sinusoidal"]:
            o = stats["rows"].get(other)
            if not o or o["seeds"] != ref["seeds"]:
                continue
            res = {}
            for kind, levels in GEOM_LEVELS.items():
                for lv in levels:
                    k = f"{kind}:{lv}"
                    d = ((np.array(o["clean"]["values"]) - np.array(o[k]["values"]))
                         - (np.array(ref["clean"]["values"]) - np.array(ref[k]["values"])))
                    res[k] = _paired(d)
            stats["paired_drop_vs_2drope"][other] = res
            print(f"  drop({other}) - drop(2D-RoPE): "
                  + "  ".join(f"{k.split(':')[1]}:{v['mean']:+.2f}[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
                              for k, v in res.items()))
    # Factorial: effect of 2D cross keys on the drop, seed-blocked over both splits.
    cells = {lab: stats["rows"].get(lab) for lab, _ in GEOM_ROWS[7:]}
    if all(cells.values()):
        res = {}
        for kind, levels in GEOM_LEVELS.items():
            for lv in levels:
                k = f"{kind}:{lv}"
                drop = {lab: np.array(e["clean"]["values"]) - np.array(e[k]["values"]) for lab, e in cells.items()}
                none = (drop["$32{:}32$ + no cross"] + drop["$48{:}16$ + no cross"]) / 2
                cross = (drop["$32{:}32$ + 2D cross"] + drop["$48{:}16$ + 2D cross"]) / 2
                res[k] = _paired(none - cross)
        stats["cross_keys_drop_reduction"] = res
        print("  drop(no cross) - drop(2D cross): "
              + "  ".join(f"{k.split(':')[1]}:{v['mean']:+.2f}[{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]"
                          for k, v in res.items()))
    print(f"geom: {len(stats['rows'])} configs in {geo_dir}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "geom.json"), "w") as f:
        json.dump(stats, f, indent=1)
    for kind in GEOM_LEVELS:
        with open(os.path.join(out_dir, f"geom_{kind}_rows.tex"), "w") as f:
            f.write("% generated by code/make_tables.py -- do not edit\n" + "\n".join(rows[kind]) + "\n")
    for label, e in stats["rows"].items():
        print(f"  {label:22s} n={len(e['seeds'])} clean {e['clean']['mean']:.2f}"
              + "".join(f"  {k.split(':')[1]}:{e[k]['mean']:.2f}" for k in e if ':' in k))


def _paired(d):
    """Mean, sd, 95% CI (t, df=n-1) and two-sided p of seed-paired differences d."""
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    se = sd / math.sqrt(n)
    tc = t_crit(n - 1)
    p = t_two_sided_p(m / se, n - 1) if se > 0 else float("nan")
    return {"mean": m, "sd": sd, "ci95": [m - tc * se, m + tc * se], "p": p, "values": [float(x) for x in d]}


# ---------------------------------------------------------------- E2 cross-attention diagnostic
XATTN_METRICS = ["spread", "entropy", "backtrack", "backtrack_peak"]


def xattn(res_dir, out_dir, seeds):
    """Per-layer cross-attention profile; effect of 2D-RoPE keys = mean over both splits of
    (2D cross - no cross), paired by seed. Groups: regular / irregular, all / correct / wrong."""
    xdir = os.path.join(res_dir, "xattn")
    runs = {}
    for split in ["32-32", "48-16"]:
        for cross in ["no-cross", "2d-cross"]:
            runs[(split, cross)] = load_seeds(xdir, f"abl_factorial_{split}-{cross}", seeds)
    common = sorted(set.intersection(*[set(r) for r in runs.values()]))
    if not common:
        raise SystemExit(f"no complete seed in {xdir}")
    layers = [f"L{i}" for i in range(1, 7)]
    stats = {"seeds": common, "groups": {}}
    rows = []
    for grp in ["regular", "irregular"]:
        for sub in ["all", "correct", "wrong"]:
            g = {}
            for met in XATTN_METRICS:
                for L in layers:
                    key = f"{L}_{met}"
                    val = {c: np.array([runs[c][s]["summary"][grp][sub][key] for s in common]) for c in runs}
                    none = (val[("32-32", "no-cross")] + val[("48-16", "no-cross")]) / 2
                    cross = (val[("32-32", "2d-cross")] + val[("48-16", "2d-cross")]) / 2
                    g[key] = {"none": float(none.mean()), "cross": float(cross.mean()), "effect": _paired(cross - none)}
            stats["groups"][f"{grp}/{sub}"] = g
            for met in XATTN_METRICS:
                cells = [f"{g[f'{L}_{met}']['none']:.3f}/{g[f'{L}_{met}']['cross']:.3f}" for L in layers]
                rows.append(f"{grp} & {sub} & {met} & " + " & ".join(cells) + r" \\")
    _write(out_dir, "xattn", stats, rows)
    print(f"xattn: seeds {common}  (value: none -> 2D keys, effect [95% CI])")
    for gk in ["regular/all", "irregular/all", "irregular/wrong"]:
        g = stats["groups"][gk]
        for met in XATTN_METRICS:
            print(f"  {gk:16s} {met:15s} " + "  ".join(
                f"{L}:{g[f'{L}_{met}']['none']:.3f}->{g[f'{L}_{met}']['cross']:.3f}"
                f"({'*' if g[f'{L}_{met}']['effect']['ci95'][0] > 0 or g[f'{L}_{met}']['effect']['ci95'][1] < 0 else ' '})"
                for L in layers))


FULL_ROWS = [("2D-RoPE, seed 0, same pipeline (ours)", "full_2drope_s0"),
             ("1D RoPE, same pipeline (ours)", "full_1drope_s0"),
             ("1D sinusoidal, same pipeline (ours)", "full_1dsin_s0")]


def full(res_dir, out_dir, seeds):
    stats, rows = {}, []
    for label, stem in FULL_ROWS:
        f = os.path.join(res_dir, f"{stem}.json")
        if not os.path.exists(f):
            print(f"  [missing] {f}")
            continue
        r = json.load(open(f))
        stats[label] = r
        rows.append(f"{label} & " + " & ".join(f"{r[b]:.1f}" for b in BENCH) + f" & {r['Avg']:.2f} \\\\")
        print("  " + rows[-1])
    _write(out_dir, "full", stats, rows)


PARSEQ_ROWS = [("PARSeq (learned PE)", "parseq_learned"), ("PARSeq + 2D-RoPE", "parseq_rope2d")]


def parseq(res_dir, out_dir, seeds):
    """E5: rows per protocol (1015/2077 and 857/1811), mean +- sd over the seeds both arms finished;
    2D-RoPE - learned, paired by seed (each pair was trained on the same GPU)."""
    runs = {stem: load_seeds(res_dir, stem, seeds) for _, stem in PARSEQ_ROWS}
    common = sorted(set.intersection(*[set(r) for r in runs.values()]))
    if not common:
        raise SystemExit(f"no seed finished by both arms in {res_dir}")
    stats, rows = {"seeds": common}, []
    for proto in ["1015_2077", "857_1811"]:
        st = {}
        rows.append(f"% {proto}")
        for label, stem in PARSEQ_ROWS:
            st[label] = {}
            cells = []
            for b in BENCH + ["Avg"]:
                v = np.array([runs[stem][s][proto][b] for s in common])
                sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
                st[label][b] = {"mean": float(v.mean()), "sd": sd, "values": [float(x) for x in v]}
                cells.append(fmt(v.mean(), sd, 2 if b == "Avg" else 1))
            rows.append(f"{label} & " + " & ".join(cells) + f" & {len(common)} \\\\")
        diff = {}
        for b in BENCH + ["Avg"]:
            d = np.array([runs["parseq_rope2d"][s][proto][b] - runs["parseq_learned"][s][proto][b] for s in common])
            diff[b] = _paired(d) if len(d) > 1 else {"mean": float(d[0]), "values": [float(d[0])]}
        st["2D-RoPE - learned (paired)"] = diff
        stats[proto] = st
    _write(out_dir, "parseq", stats, rows)
    print(f"parseq: seeds {common}")
    for proto in ["1015_2077", "857_1811"]:
        a = stats[proto]
        d = a["2D-RoPE - learned (paired)"]["Avg"]
        ci = f"  95% CI [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]  p={d['p']:.3f}" if "ci95" in d else ""
        print(f"  {proto}: learned {a['PARSeq (learned PE)']['Avg']['mean']:.2f}  "
              f"2D-RoPE {a['PARSeq + 2D-RoPE']['Avg']['mean']:.2f}  diff {d['mean']:+.2f} "
              f"(per seed {', '.join(f'{x:+.2f}' for x in d['values'])}){ci}")


def _write(out_dir, name, stats, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(stats, f, indent=1)
    with open(os.path.join(out_dir, f"{name}_rows.tex"), "w") as f:
        f.write("% generated by code/make_tables.py -- do not edit\n" + "\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["factorial", "pe", "pe_fs4", "geom", "xattn", "full", "parseq"])
    ap.add_argument("--results", default="results/rev1")
    ap.add_argument("--out_dir", default="results/rev1/tables")
    ap.add_argument("--parseq_results", default="../third_party/parseq/results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    if args.study == "parseq":
        parseq(args.parseq_results, args.out_dir, args.seeds)
        return
    {"factorial": factorial, "pe": pe, "pe_fs4": pe_fs4, "geom": geom, "xattn": xattn, "full": full}[args.study](args.results, args.out_dir, args.seeds)


if __name__ == "__main__":
    main()
