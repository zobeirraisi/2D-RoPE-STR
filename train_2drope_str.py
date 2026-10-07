"""
Train 2D-RoPE-STR (or any positional-encoding variant) on the DTRB LMDB data.

Examples
--------
Smoke test (random data, CPU/GPU, a few steps):
    python train_2drope_str.py --dummy --iters 5 --batch_size 4 --eval_every 0

Quick pilot on MJSynth with the real benchmarks:
    python train_2drope_str.py --train_sources MJ --iters 30000 \
        --batch_size 256 --amp --eval_every 5000

Full reproduction (MJ + SynthText, paper protocol):
    python train_2drope_str.py --train_sources MJ ST --iters 300000 \
        --batch_size 256 --amp --feature_stride 4
"""

import argparse
import math
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import (LabelConverter, build_training_set, DummySTRDataset,
                     LmdbDataset)
from model import STRModel
from evaluate import evaluate_dataset, evaluate_all


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(prog, 1.0)))


def set_seed(seed):
    """Seed Python, NumPy and PyTorch (CPU + CUDA). Runs are then repeatable in
    data order, augmentation and initialisation, though not bit-exact on GPU
    (cuDNN/AMP kernels are not forced to be deterministic)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # PyTorch gives each worker a distinct torch seed derived from the loader
    # generator; propagate it to the `random` / NumPy streams used by the
    # augmentation and the dataset's fallback sampling.
    ws = torch.initial_seed() % 2**32
    random.seed(ws)
    np.random.seed(ws)


def atomic_save(obj, path):
    """Write to <path>.tmp, fsync, then rename: a power cut mid-save leaves the
    previous checkpoint intact instead of a truncated file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_ckpt_safe(path, map_location="cpu"):
    """Load a checkpoint; if it is unreadable (e.g. written by an older,
    non-atomic save that was interrupted), move it aside and return None."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:  # truncated / corrupt file
        bad = path + ".corrupt"
        os.replace(path, bad)
        print(f"  ! could not read {path} ({type(e).__name__}); moved to {bad}")
        return None


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def parse_args():
    ap = argparse.ArgumentParser(description="2D-RoPE-STR training")
    ap.add_argument("--data_root", default="data_lmdb_release")
    ap.add_argument("--train_sources", nargs="+", default=["MJ", "ST"])
    ap.add_argument("--limit_train", type=int, default=0,
                    help="cap #training samples (0 = use all); useful for pilots")
    ap.add_argument("--pe_type", default="2drope",
                    choices=["none", "1d_sin", "1d_learn", "2d_sin", "2d_learn",
                             "1drope", "2drope"])
    ap.add_argument("--learnable_freq", action="store_true")
    ap.add_argument("--dim_split", type=float, default=0.5)
    ap.add_argument("--rope_base_h", type=float, default=10000.0,
                    help="2D-RoPE frequency base for the row (height) axis")
    ap.add_argument("--rope_base_w", type=float, default=10000.0,
                    help="2D-RoPE frequency base for the column (width) axis")
    ap.add_argument("--cross_pe", default=None,
                    choices=["none", "1drope", "2drope"],
                    help="PE in decoder cross-attention keys (default: legacy decoder, no PE)")
    ap.add_argument("--feature_stride", type=int, default=8, choices=[4, 8, 16, 32])
    ap.add_argument("--num_encoder_layers", type=int, default=6)
    ap.add_argument("--num_decoder_layers", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=25)
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="gradient accumulation; effective batch = batch_size * accum_steps")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_aug", action="store_true", help="disable train-time augmentation")
    ap.add_argument("--dummy", action="store_true", help="train on random data")
    ap.add_argument("--eval_every", type=int, default=5000,
                    help="run benchmark eval every N iters (0 = only at the end)")
    ap.add_argument("--save_every", type=int, default=2500,
                    help="write a resumable snapshot every N iters (0 = only at eval)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (Python/NumPy/PyTorch + DataLoader workers); "
                         "omit for the legacy unseeded behaviour")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint to resume from (continues at its saved "
                         "iteration; loads optimizer/scaler state when present)")
    args = ap.parse_args()
    if args.run_name is None:
        args.run_name = f"{args.pe_type}_fs{args.feature_stride}"
    return args


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = getattr(args, "seed", None)
    if seed is not None:
        set_seed(seed)
    print(f"Device: {device} | run: {args.run_name} | seed: {seed}")
    os.makedirs(args.save_dir, exist_ok=True)

    converter = LabelConverter(max_len=args.max_len)

    if args.dummy:
        train_set = DummySTRDataset(converter, num_samples=max(args.batch_size * 4, 64))
    else:
        train_set = build_training_set(args.data_root, converter,
                                       tuple(args.train_sources),
                                       augment=not args.no_aug)
        if args.limit_train and args.limit_train < len(train_set):
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(train_set), generator=g)[: args.limit_train]
            train_set = Subset(train_set, idx.tolist())
    print(f"Training samples: {len(train_set):,}")

    loader_kw = {}
    loader_gen = None
    if seed is not None:
        loader_gen = torch.Generator().manual_seed(seed)
        loader_kw = dict(generator=loader_gen, worker_init_fn=seed_worker)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.num_workers > 0,
                        **loader_kw)

    model = STRModel(
        vocab_size=converter.vocab_size, pe_type=args.pe_type,
        learnable_freq=args.learnable_freq, dim_split=args.dim_split,
        feature_stride=args.feature_stride,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        max_len=args.max_len, pretrained_backbone=not args.dummy,
        rope_base_h=args.rope_base_h, rope_base_w=args.rope_base_w,
        cross_pe=args.cross_pe,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=converter.PAD,
                                    label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    cfg = vars(args).copy()
    last_path = os.path.join(args.save_dir, f"{args.run_name}.pt")
    best_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
    best_avg = -1.0
    start_iter = 0

    def save_ckpt(path, it, res=None):
        ck = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict(),
              "config": cfg, "iteration": it}
        if res is not None:
            ck["results"] = res
        atomic_save(ck, path)

    ck = load_ckpt_safe(args.resume, device) if args.resume else None
    if args.resume and ck is None:
        print("  ! resume checkpoint unusable -> training from scratch")
    if ck is not None:
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        else:
            print("  ! no optimizer state in checkpoint -> warm-start (fresh AdamW)")
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_iter = int(ck.get("iteration", 0))
        # don't let a later worse eval overwrite an existing, better best
        if os.path.exists(best_path):
            best_avg = torch.load(best_path, map_location="cpu",
                                  weights_only=False).get("results", {}).get("Avg", -1.0)
        print(f"Resumed from {args.resume} at iteration {start_iter} "
              f"(best Avg so far={best_avg:.2f}%)")
        if seed is not None:
            # Advance the RNG streams so a resumed run does not replay the data
            # order of iteration 0. Resumed runs are statistically equivalent to,
            # not bit-identical with, an uninterrupted run.
            set_seed(seed + start_iter)
            loader_gen.manual_seed(seed + start_iter)

    def do_eval(it):
        nonlocal best_avg
        print(f"\n[iter {it}] benchmark evaluation:")
        res = evaluate_all(model, args.data_root, converter, device,
                           batch_size=args.eval_batch_size, num_workers=args.num_workers,
                           max_len=args.max_len + 2, amp=args.amp)
        model.train()
        save_ckpt(last_path, it, res)
        if res.get("Avg", -1) > best_avg:
            best_avg = res["Avg"]
            save_ckpt(best_path, it, res)
            print(f"  -> new best Avg={best_avg:.2f}% (saved {best_path})")
        return res

    data_iter = infinite(loader)
    model.train()
    running, t0 = 0.0, time.time()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(start_iter + 1, args.iters + 1), desc="train",
                initial=start_iter, total=args.iters)
    results = {}
    for it in pbar:
        lr = cosine_lr(it, args.iters, args.lr, args.warmup)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for _ in range(args.accum_steps):
            images, tgt_in, tgt_out = next(data_iter)
            images = images.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(images, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 tgt_out.reshape(-1)) / args.accum_steps
            scaler.scale(loss).backward()
            running += loss.item() * args.accum_steps

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % args.log_every == 0:
            pbar.set_postfix(loss=f"{running / args.log_every:.4f}", lr=f"{lr:.2e}")
            running = 0.0

        if args.eval_every and it % args.eval_every == 0 and not args.dummy:
            results = do_eval(it)
        elif args.save_every and it % args.save_every == 0:
            save_ckpt(last_path, it)

    save_ckpt(last_path, args.iters)
    print(f"\nSaved checkpoint: {last_path}")
    print(f"Total wall time: {(time.time() - t0) / 60:.1f} min")

    if not args.dummy:
        print("\nFinal benchmark evaluation:")
        results = do_eval(args.iters)
        print(f"\nBest Avg over training: {best_avg:.2f}%  (checkpoint: {best_path})")
    return results


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
