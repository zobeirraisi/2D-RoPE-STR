# 2D-RoPE-STR

Official implementation of **2D Rotary Position Embeddings for Scene Text Recognition**.

## Results

**90.40% average accuracy** on six standard benchmarks (MJ + SynthText train protocol).

| Benchmark | Accuracy |
|-----------|----------|
| IIIT5k    | 93.2%    |
| SVT       | 91.5%    |
| IC13      | 97.0%    |
| IC15      | 84.1%    |
| CUTE80    | 90.3%    |
| SVTP      | 86.2%    |

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/zobeirraisi/2D-RoPE-STR.git
cd 2D-RoPE-STR

# Install dependencies
pip install torch torchvision lmdb pillow tqdm
```

### Evaluate with Pretrained Checkpoint

```bash
# Download checkpoint (link coming once paper is published)
python evaluate.py --checkpoint checkpoints/2drope_baseline.pt \
    --out results/eval.json
```

### Train from Scratch

**Pilot (fast, ~1 hour on RTX 4090):**
```bash
python train_2drope_str.py \
    --train_sources MJ \
    --iters 30000 \
    --batch_size 256 \
    --amp \
    --feature_stride 8 \
    --run_name pilot_2drope
```

**Full reproduction (paper protocol, days on RTX 4090):**
```bash
python train_2drope_str.py \
    --train_sources MJ ST \
    --iters 300000 \
    --batch_size 256 \
    --amp \
    --feature_stride 4 \
    --run_name full_2drope
```

**Smoke test (verify setup, seconds):**
```bash
python train_2drope_str.py --dummy --iters 5 --batch_size 4
```

## Dataset Setup

Download the Deep-Text-Recognition-Benchmark (DTRB) LMDB datasets and organize as:

```
data_lmdb_release/
├── training/
│   ├── MJ/{train,val,test}/
│   └── ST/
└── evaluation/
    ├── IIIT5k_3000/
    ├── SVT/
    ├── IC13_1015/
    ├── IC15_2077/
    ├── CUTE80/
    └── SVTP/
```

See `dataset.py` for LMDB format details.

## Code Structure

| File | Purpose |
|------|---------|
| `model.py` | `STRModel`: ResNet-50 → Transformer encoder (pluggable PE) → autoregressive decoder |
| `train_2drope_str.py` | Training driver with AMP, cosine LR schedule, checkpointing, periodic eval |
| `evaluate.py` | Benchmark evaluation on all 6 datasets; outputs JSON results |
| `run_ablations.py` | Positional encoding type / layers / frequency / cross-attention ablations |
| `dataset.py` | LMDB reader, `LabelConverter`, transforms, dummy dataset |
| `make_qualitative.py` | Generate visualizations and attention maps |

## Ablations

Reproduce the ablation studies from the paper:

```bash
# Positional encoding type (none, 1d_sin, 1d_learn, 2d_sin, 2d_learn, 1drope, 2drope)
python run_ablations.py --study pe --iters 30000 --train_sources MJ --out results/ablation_pe.json

# Encoder depth (3, 6, 9, 12 layers)
python run_ablations.py --study layers --iters 30000 --train_sources MJ --out results/ablation_layers.json

# Frequency component scaling
python run_ablations.py --study freq --iters 30000 --train_sources MJ --out results/ablation_freq.json

# Cross-attention variants
python run_ablations.py --study cross --iters 30000 --train_sources MJ --out results/ablation_cross.json
```

## Paper

[Link coming once published on Pattern Recognition Letters]

## Citation

```bibtex

```

## License

MIT

## Reproducibility

- Random seed fixed in training (see `train_2drope_str.py --seed`)
- AMP-safe (tested on RTX 4090; CPU fallback available)
- Checkpoint resumption preserves optimizer/scaler state
- LMDB caching compatible with multi-GPU (via `lmdb_readonly_wrapper`)
