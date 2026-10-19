# 2D-RoPE-STR

Code for **"2D Rotary Position Embedding for Scene Text Recognition with Transformers"**
(Z. Raisi, under review at IJDAR).

The paper is a controlled study of positional encoding in an encoder-decoder scene text
recognizer (ResNet-50 → Transformer encoder with a pluggable positional encoding →
autoregressive decoder). Only the positional encoding changes between compared models.

## Results of the released checkpoint

Model of Table 2 in the paper: 2D-RoPE in encoder self-attention, equal row/column split
(32:32 per head), learnable per-axis frequency scales, output stride 4 (8×32 grid), standard
decoder without positional encoding in cross-attention, trained 300k iterations on MJ+ST.

| IIIT5K | SVT | IC13-1015 | IC15-2077 | CUTE80 | SVTP | Avg |
|---|---|---|---|---|---|---|
| 96.17 | 92.43 | 95.17 | 81.17 | 90.97 | 86.51 | 90.40 |

On IC13-857 / IC15-1811: 97.08 / 84.87 (Avg 91.34).
Word accuracy, case-insensitive 36-character set, greedy decoding.

> An earlier version of this README listed different accuracies, a different default command,
> and a different venue. Those accuracies did not correspond to the released checkpoint; the
> table above is the output of `evaluate.py` on it.

## Installation

```bash
git clone https://github.com/zobeirraisi/2D-RoPE-STR.git
cd 2D-RoPE-STR
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.7.1 (CUDA 11.8), on an RTX 4090 and an RTX 3090.

## Checkpoints

The four full-schedule models of Table 2 are attached to the GitHub release
[`v1.0-ijdar-rev1`](https://github.com/zobeirraisi/2D-RoPE-STR/releases/tag/v1.0-ijdar-rev1)
(weights only, fp32, about 265 MB each; optimizer state removed, weights unchanged):

| File | Model | Avg |
|---|---|---|
| `full_2drope_best.pt` | 2D-RoPE, the model of Table 2 (unseeded) | 90.40 |
| `full_2drope_s0.pt` | 2D-RoPE, seed 0 | 90.38 |
| `full_1drope_s0.pt` | 1D RoPE, seed 0 | 89.73 |
| `full_1dsin_s0.pt` | 1D sinusoidal, seed 0 | 89.90 |

```bash
BASE=https://github.com/zobeirraisi/2D-RoPE-STR/releases/download/v1.0-ijdar-rev1
mkdir -p release && cd release
for f in full_2drope_best.pt full_2drope_s0.pt full_1drope_s0.pt full_1dsin_s0.pt SHA256SUMS; do
  wget $BASE/$f
done
sha256sum -c SHA256SUMS && cd ..
mkdir -p checkpoints checkpoints_rev1   # paths used by the commands below
mv release/full_2drope_best.pt checkpoints/ && mv release/full_*_s0.pt checkpoints_rev1/
```

Each file holds `model` (state dict), `config` (the training arguments, read by `evaluate.py`),
`iteration`, and `results` (the evaluation output below).

## Data

DTRB LMDB release, organized as

```
data_lmdb_release/
├── training/{MJ/{train,val,test}, ST}
└── evaluation/{IIIT5k_3000, SVT, IC13_1015, IC13_857, IC15_2077, IC15_1811, CUTE80, SVTP}
```

Evaluation counts every image whose label, after lower-casing and removing characters outside
the 36-character set, is non-empty and at most 25 characters long (all images of the six
benchmarks qualify). Training samples that do not qualify are replaced by random others.

## Reproduce the main model

```bash
python train_2drope_str.py \
    --train_sources MJ ST --iters 300000 \
    --pe_type 2drope --learnable_freq --dim_split 0.5 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 \
    --amp --seed 0 --run_name full_2drope
python evaluate.py --checkpoint checkpoints/full_2drope_best.pt --out results/full_2drope.json
```

The defaults of `train_2drope_str.py` (`--feature_stride 8`, no `--learnable_freq`,
`--batch_size 256`) are the ablation settings, not the main-model settings. The released
checkpoint was trained before seeds were fixed, so a re-run reproduces it statistically, not
bit for bit.

Full-schedule rows of Table 2: a seeded re-run of the main configuration and the 1D baselines
(same recipe; the 1D models use fixed frequency bases):

```bash
python train_2drope_str.py --pe_type 2drope --learnable_freq --dim_split 0.5 --train_sources MJ ST \
    --iters 300000 --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 --run_name full_2drope_s0
python train_2drope_str.py --pe_type 1drope --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 --run_name full_1drope_s0
python train_2drope_str.py --pe_type 1d_sin --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 --run_name full_1dsin_s0
```

Evaluation output (`evaluate.py`, IC13-1015 / IC15-2077, single runs):

| Model | IIIT5K | SVT | IC13 | IC15 | CUTE80 | SVTP | Avg |
|---|---|---|---|---|---|---|---|
| 2D-RoPE, seed 0 (`full_2drope_s0`) | 96.17 | 92.89 | 94.98 | 81.13 | 90.28 | 86.82 | 90.38 |
| 1D RoPE, seed 0 (`full_1drope_s0`) | 96.20 | 92.58 | 95.07 | 81.22 | 88.19 | 85.12 | 89.73 |
| 1D sinusoidal, seed 0 (`full_1dsin_s0`) | 96.00 | 91.65 | 94.98 | 80.89 | 90.28 | 85.58 | 89.90 |

## Reproduce the controlled comparisons (60k iterations, MJ+ST, seeds 0–2)

```bash
for s in 0 1 2; do
  # Table 3: seven positional encodings (stride 8, 4x16 grid)
  python run_ablations.py --study pe --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed $s --save_dir checkpoints_rev1 --resume \
      --out results/rev1/abl_pe_s$s.json
  # Table 6: 2x2 factorial, row/column split x RoPE on cross-attention keys
  python run_ablations.py --study factorial --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed $s --save_dir checkpoints_rev1 --resume \
      --out results/rev1/abl_factorial_s$s.json
done

# Table 5: 1D vs 2D RoPE with an 8x32 grid (stride 4), seeds 0-2
for s in 0 1 2; do
  python run_ablations.py --study pe_fs4 --iters 60000 --feature_stride 4 \
      --batch_size 128 --accum_steps 2 --amp --train_sources MJ ST --seed $s \
      --save_dir checkpoints_rev1 --resume --out results/rev1/abl_pe_fs4_s$s.json
done

# Table 5 (4x16 block): 2D-RoPE with fixed frequencies (no learnable scales), seeds 0-2
for s in 0 1 2; do
  python run_ablations.py --study pe_fixfreq --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed $s --save_dir checkpoints_rev1 --resume \
      --out results/rev1/abl_pe_fixfreq_s$s.json
done
```

Always pass `--train_sources MJ ST` and `--seed`.

Single-run ablations (Table 3, single-run column; Table 4 (a)–(d)). These runs predate `--seed`,
so a re-run reproduces them statistically, not exactly; Table 4 (e) collects the runs that share a
configuration. Same schedule (60k iterations, stride 8, batch 256, AMP):

```bash
for study in pe freq dim layers cross; do
  python run_ablations.py --study $study --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --out results/abl_$study.json
done
```

The `cross` study (Table 4 (d)) uses the pre-norm decoder that supports rotated keys in all three rows.

Robustness to controlled rotation / perspective (Fig. 3) and the cross-attention diagnostic:

```bash
python eval_geometric.py --data_root data_lmdb_release --out_dir results/rev1/geom \
    --checkpoints checkpoints_rev1/abl_pe_*_s[0-9].pt checkpoints_rev1/abl_factorial_*_s[0-9].pt
python eval_geometric.py --data_root data_lmdb_release --out_dir results/rev1/geom \
    --checkpoints checkpoints_rev1/full_1drope_s0.pt checkpoints_rev1/full_1dsin_s0.pt \
                  checkpoints_rev1/full_2drope_s0.pt checkpoints/full_2drope_best.pt
python analyze_cross_attention.py --data_root data_lmdb_release --out_dir results/rev1/xattn \
    --checkpoints checkpoints_rev1/abl_factorial_*_s[0-9].pt
```

## Tables and figures

The multi-seed tables, the full-schedule rows, and Figs. 1 and 3 are generated from the result
JSONs (baseline rows of Table 2 are copied from the cited papers, with table numbers). The JSONs are
in this repository: `results/` (single runs), `results/rev1/` (multi-seed and full-schedule runs),
`results/rev1/geom/` (the full geometric sweep, including the encodings and factorial models not
drawn in Fig. 3), `results/rev1/xattn/` (per-layer cross-attention values), and `results/rev1/parseq/`
(PARSeq runs):

```bash
python make_tables.py pe          # Table 3 multi-seed column + paired differences to 2D-RoPE
python make_tables.py pe_fs4      # Table 5 + paired difference and pre-specified decision rule
python make_tables.py pe_fixfreq  # Table 5 (4x16 block) + paired differences (fixed 2D-RoPE vs 1D RoPE / learned)
python make_tables.py factorial   # Table 6 + main effects / interaction (seed-blocked)
python make_tables.py geom        # Fig. 3 data + paired differences in accuracy drop + full-schedule rows
python make_tables.py xattn       # per-layer cross-attention profile
python make_tables.py full        # full-schedule rows of Table 2 + per-benchmark differences
python make_tables.py parseq      # Table 7 (reads results/rev1/parseq)
python make_overview_figure.py    # Fig. 1
python make_geom_figure.py        # Fig. 3
```

Disagreement analysis (Table 8), the disagreement examples (Fig. 4) and the attention maps (Fig. 5):

```bash
python make_qualitative.py        # Table 8 counts -> results/netwins.json
                                  # (single-run 1D-sinusoidal / 2D-RoPE pair of Table 3)
python make_tables.py netwins     # Table 8 rows + McNemar p -> results/rev1/tables/netwins_rows.tex
python make_disagreement_figure.py   # Fig. 4: first disagreement in each direction on CUTE80, SVTP, IC15
                                     # (rule fixed in advance) -> results/rev1/fig4_disagreements.json
python make_attention_sbs.py --out fig_attention_sbs.pdf --models \
    "1D sinusoidal=checkpoints_rev1/full_1dsin_s0.pt" "1D RoPE=checkpoints_rev1/full_1drope_s0.pt" \
    "2D-RoPE=checkpoints_rev1/full_2drope_s0.pt"   # Fig. 5 (seeded full-schedule models, 8x32 grid)
```

Fig. 4 was laid out in draw.io from the crops written by `make_qualitative.py`.

Outputs go to `results/rev1/tables/`.

The PARSeq transfer experiment (Table 7) uses a modified copy of the official PARSeq code with a
2D-RoPE option for its ViT encoder (`model.pos_enc=rope2d`), trained on the same MJ+ST LMDBs with
the official synthetic-data recipe on one GPU (2-step gradient accumulation, learning rate rescaled
to the 2-GPU step size). The modification is `patches/parseq_rope2d.patch`, which applies to
upstream PARSeq commit `1902db0`; `scripts/e5_parseq.sh` has the exact training and evaluation
commands:

```bash
git clone https://github.com/baudm/parseq.git ../third_party/parseq
git -C ../third_party/parseq checkout 1902db0 && git -C ../third_party/parseq apply "$PWD/patches/parseq_rope2d.patch"
bash scripts/e5_parseq.sh pc rope2d:0 learned:0   # seeds 1 and 2 likewise
```

Mean ± sd over seeds 0–2 (IC13-1015 / IC15-2077):

| Model | IIIT5K | SVT | IC13 | IC15 | CUTE80 | SVTP | Avg |
|---|---|---|---|---|---|---|---|
| PARSeq (learned PE) | 96.28 ± 0.11 | 92.84 ± 0.32 | 95.73 ± 0.15 | 82.47 ± 0.25 | 91.20 ± 0.87 | 87.91 ± 0.81 | 91.07 ± 0.25 |
| PARSeq + 2D-RoPE | 96.71 ± 0.27 | 92.89 ± 0.56 | 95.93 ± 0.50 | 83.26 ± 0.28 | 93.06 ± 0.69 | 89.04 ± 0.09 | 91.81 ± 0.19 |

## Code structure

| File | Purpose |
|------|---------|
| `model.py` | `STRModel` and all positional encodings (`none`, `1d_sin`, `1d_learn`, `2d_sin`, `2d_learn`, `1drope`, `2drope`; optional RoPE on cross-attention keys) |
| `train_2drope_str.py` | Training (AMP, cosine schedule, gradient accumulation, seeding, atomic resumable checkpoints) |
| `evaluate.py` | Word accuracy on the six benchmarks, JSON output |
| `run_ablations.py` | Studies: `pe`, `dim`, `freq`, `layers`, `cross`, `pe_fs4`, `factorial` |
| `eval_geometric.py` | Controlled rotation / perspective evaluation |
| `analyze_cross_attention.py` | Per-layer decoder cross-attention diagnostic |
| `make_tables.py`, `make_*_figure.py` | Tables and figures from the result JSONs |
| `make_qualitative.py`, `make_attention_sbs.py` | Qualitative examples, disagreement counts, attention maps |
| `dataset.py` | LMDB reader, label converter, transforms |
| `scripts/` | Power-loss-safe launchers used for the revision experiments |
| `patches/parseq_rope2d.patch` | 2D-RoPE option for PARSeq (transfer experiment, Table 7) |

## Reproducibility

- `--seed` fixes PyTorch, NumPy, Python, and DataLoader worker seeds. GPU kernels are not forced
  to be deterministic, so seeded runs are repeatable in data order and initialization, not bit
  for bit. Runs made before the revision did not fix the seed; the paper reports their
  run-to-run spread.
- AdamW weight decay (0.01) applies to all parameters, including the learnable frequency scales.
- Checkpoint resumption restores optimizer and scaler state.

## License

MIT
