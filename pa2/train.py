"""Train one model under the shared protocol.

Adam(lr=1e-3, no schedule), batch 32, 30 epochs, final-epoch weights kept,
one random window per training video per epoch, 9-step rollout without
teacher forcing, loss averaged over pixels and valid target frames of each
video, then over the videos in the batch.

Usage:  python -m pa2.train --model A --seed 0
        python -m pa2.train --model D --arch D_k5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from .data import (AUG_STARTS, NORMAL_STARTS, apply_g, batch_window_indices, default_device,
                   ensure_dir, load_moving_mnist, split, sync, get_window)
from .models import build, config_dict, n_params

BATCH = 32
LR = 1e-3

# model -> (architecture, speed regime, symmetry augmentation)
MODEL_TABLE = {
    "A": ("A", "normal", False),
    "B": ("A", "normal", True),
    "C": ("A", "speed", False),
    "D": (None, "speed", False),    # architecture chosen on validation
    "DB": (None, "speed", True),    # optional D+B
}


def sample_windows(n: int, regime: str, gen: torch.Generator):
    """Draw (t, k) for n videos. Speed regime: k ~ U{1,2,3}, then t ~ U(allowed starts for k)."""
    if regime == "normal":
        starts = torch.tensor(NORMAL_STARTS[1])
        t = starts[torch.randint(len(starts), (n,), generator=gen)]
        k = torch.ones(n, dtype=torch.long)
        return t, k
    k = torch.randint(1, 4, (n,), generator=gen)
    t = torch.empty(n, dtype=torch.long)
    for kk in (1, 2, 3):
        sel = k == kk
        starts = torch.tensor(AUG_STARTS[kk])
        t[sel] = starts[torch.randint(len(starts), (int(sel.sum()),), generator=gen)]
    return t, k


def masked_video_mse(pred, Y, mask):
    """Squared error averaged over pixels and valid target frames per video, then over videos."""
    se = (pred - Y).pow(2).mean(dim=(2, 3, 4))          # (B, 5) per-frame MSE
    m = mask.to(se.dtype)
    per_video = (se * m).sum(1) / m.sum(1)
    return per_video.mean()


@torch.no_grad()
def val_error(model, val_u8, device, k=1, batch=100):
    """Mean E(5)-style error on validation at speed k, t=1 (valid targets only)."""
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(val_u8), batch):
        v = val_u8[s:s + batch].to(device).float().div_(255.0)
        X, Y, mask = get_window(v, 1, k)
        pred = model(X)
        se = (pred - Y).pow(2).mean(dim=(2, 3, 4))[:, mask.to(device)]
        tot += se.mean(1).sum().item()
        n += len(v)
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_TABLE))
    ap.add_argument("--arch", default=None, help="architecture for D / DB (see models.CONFIGS)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default=None, help="run name (default: model[_arch]_s{seed})")
    ap.add_argument("--max-batches", type=int, default=None, help="smoke test only")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="checkpoints", help="checkpoint dir (logs go to results/train_logs "
                    "unless --out is changed, then next to the checkpoint)")
    ap.add_argument("--val-speeds", action="store_true", help="also log final val error at 2x/3x")
    args = ap.parse_args()

    arch, regime, sym = MODEL_TABLE[args.model]
    if arch is None:
        assert args.arch is not None, "--arch required for D / DB"
        arch = args.arch
    tag = args.tag or (f"{args.model}_s{args.seed}" if args.model in ("A", "B", "C")
                       else f"{args.model}_{arch}_s{args.seed}")
    device = default_device()

    data = load_moving_mnist(args.data_root)
    train_u8, val_u8, _ = split(data)       # test split never touched here
    train_dev = train_u8.to(device)          # uint8 on device (655 MB)

    torch.manual_seed(args.seed)             # initialisation
    gen = torch.Generator().manual_seed(args.seed)  # shuffling, windows, augmentation
    model = build(arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[{tag}] arch={arch} regime={regime} sym={sym} params={n_params(model)} device={device}",
          flush=True)

    log = {"tag": tag, "model": args.model, "arch": arch, "config": config_dict(arch),
           "regime": regime, "sym_aug": sym, "seed": args.seed, "params": n_params(model),
           "device": str(device), "machine": platform.platform(), "epochs": [],
           "torch": torch.__version__}
    N = len(train_u8)
    for ep in range(1, args.epochs + 1):
        sync(device)
        t0 = time.perf_counter()
        perm = torch.randperm(N, generator=gen)
        t_all, k_all = sample_windows(N, regime, gen)
        g_all = torch.randint(8, (N,), generator=gen) if sym else torch.zeros(N, dtype=torch.long)
        run_loss, nb = 0.0, 0
        for b, s in enumerate(range(0, N, BATCH)):
            if args.max_batches and b >= args.max_batches:
                break
            ids = perm[s:s + BATCH]
            idx, mask = batch_window_indices(t_all[ids], k_all[ids])
            vids = train_dev[ids.to(device)[:, None], idx.to(device)]   # (B, 10, 1, 64, 64)
            vids = vids.float().div_(255.0)
            if sym:
                g_b = g_all[ids]
                for g in g_b.unique().tolist():
                    sel = (g_b == g).to(device)
                    vids[sel] = apply_g(vids[sel], g)
            X, Y = vids[:, :5], vids[:, 5:] * mask.to(device)[:, :, None, None, None]
            pred = model(X)
            loss = masked_video_mse(pred, Y, mask.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nb += 1
        sync(device)
        ep_time = time.perf_counter() - t0
        v1 = val_error(model, val_u8, device, k=1)
        rec = {"epoch": ep, "train_loss": run_loss / nb, "val_E5_1x": v1, "epoch_time_s": ep_time}
        log["epochs"].append(rec)
        print(f"[{tag}] ep {ep:2d} loss {rec['train_loss']:.5f} val1x {v1:.5f} time {ep_time:.1f}s",
              flush=True)

    log["mean_epoch_time_s"] = float(np.mean([e["epoch_time_s"] for e in log["epochs"]]))
    if args.val_speeds:
        log["val_final"] = {f"E2_{k}x": val_error_e2(model, val_u8, device, k) for k in (1, 2, 3)}
        log["val_final"]["E5_1x"] = log["epochs"][-1]["val_E5_1x"]
        print(f"[{tag}] val final {log['val_final']}", flush=True)
    log_dir = "results/train_logs" if args.out == "checkpoints" else args.out
    ensure_dir(args.out)
    ensure_dir(log_dir)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "config": config_dict(arch)},
               os.path.join(args.out, f"{tag}.pt"))
    with open(os.path.join(log_dir, f"{tag}.json"), "w") as f:
        json.dump(log, f, indent=1)


@torch.no_grad()
def val_error_e2(model, val_u8, device, k, batch=100):
    """Validation E(2) (first two horizons) at speed k, t=1."""
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(val_u8), batch):
        v = val_u8[s:s + batch].to(device).float().div_(255.0)
        X, Y, _ = get_window(v, 1, k)
        pred = model(X)
        tot += (pred[:, :2] - Y[:, :2]).pow(2).mean(dim=(1, 2, 3, 4)).sum().item()
        n += len(v)
    model.train()
    return tot / n


if __name__ == "__main__":
    main()
