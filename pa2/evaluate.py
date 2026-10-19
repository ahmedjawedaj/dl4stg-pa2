"""Test-set metrics: horizon MSEs, E(5)/E(2), consistency gaps C_T and transformed errors E_T,
speed tests, and inference timing. All test numbers use the 1,000 test videos at t = 1."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .data import get_window, mirror, rotate, sync
from .models import Blank, CopyLast, ConvLSTMForecaster, NetConfig, Symmetrised, n_params

TRANSFORMS = {"M": mirror, "R": lambda x: rotate(x, 1)}


def load_model(tag: str, ckpt_dir: str = "checkpoints", device="cpu") -> ConvLSTMForecaster:
    ck = torch.load(os.path.join(ckpt_dir, f"{tag}.pt"), map_location="cpu")
    model = ConvLSTMForecaster(NetConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def get_predictor(name: str, device, ckpt_dir="checkpoints"):
    """'Blank', 'CopyLast', a checkpoint tag, or 'sym:<tag>' for the symmetrised wrapper."""
    if name == "Blank":
        return Blank().to(device)
    if name == "CopyLast":
        return CopyLast().to(device)
    if name.startswith("sym:"):
        return Symmetrised(load_model(name[4:], ckpt_dir, device)).eval()
    return load_model(name, ckpt_dir, device)


def test_windows(test_u8: torch.Tensor, k: int):
    v = test_u8.float().div_(255.0)
    return get_window(v, 1, k)


@torch.no_grad()
def predict(model, X: torch.Tensor, device, batch: int = 32) -> torch.Tensor:
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        out.append(model(X[s:s + batch].to(device)).float().cpu())
    return torch.cat(out)


def horizon_mse(pred: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    """(N, 5) per-video, per-horizon MSE over the 64x64 pixels."""
    return (pred - Y).pow(2).mean(dim=(2, 3, 4)).double().numpy()


def evaluate(model, test_u8, device, parts=("1x",), batch=32):
    """Return per-video arrays for the requested parts.

    '1x'    : mse_1x (N, 5)
    'M','R' : C_T and E_T per video (mean over 5 horizons and pixels), at 1x, t=1
    'speed' : mse_2x (N, 5), mse_3x (N, 2)  (valid targets only)
    """
    res = {}
    X1, Y1, _ = test_windows(test_u8, 1)
    need_base = any(p in parts for p in ("1x", "M", "R"))
    if need_base:
        P1 = predict(model, X1, device, batch)
        res["mse_1x"] = horizon_mse(P1, Y1)
    for T in ("M", "R"):
        if T in parts:
            fwd = TRANSFORMS[T]
            PT = predict(model, fwd(X1), device, batch)                 # f(T(X))
            res[f"C_{T}"] = horizon_mse(PT, fwd(P1)).mean(1)          # vs T(f(X))
            res[f"E_{T}"] = horizon_mse(PT, fwd(Y1)).mean(1)          # vs T(Y)
    if "speed" in parts:
        for k in (2, 3):
            Xk, Yk, mk = test_windows(test_u8, k)
            Pk = predict(model, Xk, device, batch)
            res[f"mse_{k}x"] = horizon_mse(Pk[:, mk], Yk[:, mk])
    return res


def summarise(res: dict) -> dict:
    """Scalars: MSE_h, E5_1x, E2 at each speed, C/E means."""
    s = {}
    if "mse_1x" in res:
        m = res["mse_1x"].mean(0)
        for h in range(5):
            s[f"MSE{h + 1}"] = float(m[h])
        s["E5_1x"] = float(res["mse_1x"].mean())
        s["E2_1x"] = float(res["mse_1x"][:, :2].mean())
    for key in ("C_M", "E_M", "C_R", "E_R"):
        if key in res:
            s[key] = float(res[key].mean())
    if "mse_2x" in res:
        s["E5_2x"] = float(res["mse_2x"].mean())
        s["E2_2x"] = float(res["mse_2x"][:, :2].mean())
        s["E2_3x"] = float(res["mse_3x"][:, :2].mean())
    return s


@torch.no_grad()
def inference_time(model, test_u8, device, batch=32, repeats=3) -> float:
    """Wall-clock seconds to predict all 1,000 original (1x, t=1) test videos at batch 32,
    under eval() and no_grad(), timed after the batches are on the device. Median of repeats
    (after one warm-up pass)."""
    X1, _, _ = test_windows(test_u8, 1)
    batches = [X1[s:s + batch].to(device) for s in range(0, len(X1), batch)]
    model.eval()
    for xb in batches[:3]:
        model(xb)
    sync(device)
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        for xb in batches:
            model(xb)
        sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def save_eval(name: str, res: dict, extra: dict | None = None, out="results/eval"):
    os.makedirs(out, exist_ok=True)
    fname = name.replace(":", "_")
    np.savez_compressed(os.path.join(out, f"{fname}.npz"), **res)
    s = summarise(res)
    if extra:
        s.update(extra)
    path = os.path.join(out, f"{fname}.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(s)
    json.dump(old, open(path, "w"), indent=1)
    return old


def fmt(v: float, sci: bool = False) -> str:
    """Three significant figures, scientific notation below 1e-3 (or always if sci)."""
    if v is None:
        return "-"
    if v == 0 and not sci:
        return "0"
    if sci or abs(v) < 1e-3:
        return f"{v:.2e}"
    r = float(f"{v:.3g}")
    if abs(r) >= 1000:
        return f"{r:.0f}"
    return f"{r:#.3g}"


def fmt_seeds(vals, sci: bool = False) -> str:
    """'mean (seed 0, seed 1)'."""
    vals = list(vals)
    if len(vals) == 1:
        return fmt(vals[0], sci)
    return f"{fmt(float(np.mean(vals)), sci)} ({fmt(vals[0], sci)}, {fmt(vals[1], sci)})"


def count_params(name: str) -> int:
    if name in ("Blank", "CopyLast"):
        return 0
    m = load_model(name.replace("sym:", ""))
    return n_params(m)
