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

Full-schedule 1D baselines of Table 2 (same recipe, fixed frequency bases):

```bash
python train_2drope_str.py --pe_type 1drope --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 --run_name full_1drope_s0
python train_2drope_str.py --pe_type 1d_sin --train_sources MJ ST --iters 300000 \
    --feature_stride 4 --batch_size 128 --accum_steps 2 --amp --seed 0 --run_name full_1dsin_s0
```

## Reproduce the controlled comparisons (60k iterations, MJ+ST, seeds 0–2)

```bash
for s in 0 1 2; do
  # Table 3: seven positional encodings (stride 8, 4x16 grid)
  python run_ablations.py --study pe --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed $s --save_dir checkpoints_rev1 --resume \
      --out results/rev1/abl_pe_s$s.json
  # Table 5: 2x2 factorial, row/column split x RoPE on cross-attention keys
  python run_ablations.py --study factorial --iters 60000 --feature_stride 8 --amp \
      --train_sources MJ ST --seed $s --save_dir checkpoints_rev1 --resume \
      --out results/rev1/abl_factorial_s$s.json
done

# 1D vs 2D RoPE with an 8x32 grid (stride 4), seeds 0-1
for s in 0 1; do
  python run_ablations.py --study pe_fs4 --iters 60000 --feature_stride 4 \
      --batch_size 128 --accum_steps 2 --amp --train_sources MJ ST --seed $s \
      --save_dir checkpoints_rev1 --resume --out results/rev1/abl_pe_fs4_s$s.json
done
```

Always pass `--train_sources MJ ST` and `--seed`.

Robustness to controlled rotation / perspective (Fig. 3) and the cross-attention diagnostic:

```bash
python eval_geometric.py --data_root data_lmdb_release --out_dir results/rev1/geom \
    --checkpoints checkpoints_rev1/abl_pe_*_s[0-9].pt checkpoints_rev1/abl_factorial_*_s[0-9].pt
python analyze_cross_attention.py --data_root data_lmdb_release --out_dir results/rev1/xattn \
    --checkpoints checkpoints_rev1/abl_factorial_*_s[0-9].pt
```

## Tables and figures

The multi-seed tables, the full-schedule rows, and Figs. 1 and 3 are generated from the result
JSONs (baseline rows of Table 2 are copied from the cited papers, with table numbers):

```bash
python make_tables.py pe          # Table 3 multi-seed column + paired differences to 2D-RoPE
python make_tables.py factorial   # Table 5 + main effects / interaction (seed-blocked)
python make_tables.py geom        # Fig. 3 data + paired differences in accuracy drop
python make_tables.py xattn       # per-layer cross-attention profile
python make_tables.py full        # full-schedule rows of Table 2
python make_overview_figure.py    # Fig. 1
python make_geom_figure.py        # Fig. 3
```

Outputs go to `results/rev1/tables/`.

The PARSeq transfer experiment uses a modified copy of the official PARSeq code with a 2D-RoPE
option for its ViT encoder (`model.pos_enc=rope2d`); see `scripts/e5_parseq.sh`.

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

## Reproducibility

- `--seed` fixes PyTorch, NumPy, Python, and DataLoader worker seeds. GPU kernels are not forced
  to be deterministic, so seeded runs are repeatable in data order and initialization, not bit
  for bit. Runs made before the revision did not fix the seed; the paper reports their
  run-to-run spread.
- AdamW weight decay (0.01) applies to all parameters, including the learnable frequency scales.
- Checkpoint resumption restores optimizer and scaler state.

## License

MIT
