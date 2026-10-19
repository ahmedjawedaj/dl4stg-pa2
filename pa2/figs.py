"""Figure 3/5/6 renderers shared by the phase 2 and phase 3 scripts."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import TEST_IDX, frames_separate, window_frames
from .evaluate import TRANSFORMS, fmt, predict, test_windows
from .plots import four_row_gif, four_row_panel, speed_panel, stack_pngs


def figure3(model, model_label, test_u8, picks, device, prefix, gif_dir="gifs", fig_dir="figures"):
    """picks: list of (T, test_position, C_value_for_title). Writes per-panel PNGs, a stacked PNG
    and one GIF per panel. Returns (stacked_png, gif_paths, per-panel stats)."""
    pngs, gifs, stats = [], [], []
    for T, i, _ in picks:
        fwd = TRANSFORMS[T]
        X, Y, _ = test_windows(test_u8[i:i + 1], 1)
        fX = predict(model, X, device)
        fTX = predict(model, fwd(X), device)
        TX, TY, TfX = fwd(X), fwd(Y), fwd(fX)
        C = ((fTX - TfX) ** 2).mean().item()
        E = ((fTX - TY) ** 2).mean().item()
        title = (f"{model_label}, T = {T}, test video {TEST_IDX.start + i} (test #{i}): "
                 f"per-video C_{T} = {fmt(C)}, E_{T} = {fmt(E)}")
        fig = four_row_panel(X[0], fX[0], TX[0], TY[0], fTX[0], TfX[0], T, title)
        p = f"{fig_dir}/{prefix}_{T}_v{i}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        pngs.append(p)
        g = f"{gif_dir}/{prefix}_{T}_v{i}.gif"
        four_row_gif(g, X[0], fX[0], TX[0], TY[0], fTX[0], TfX[0], T, title)
        gifs.append(g)
        stats.append({"T": T, "test_pos": i, "dataset_idx": TEST_IDX.start + i, "C": C, "E": E})
    out = f"{fig_dir}/{prefix}.png"
    stack_pngs(pngs, out)
    return out, gifs, stats


def displayed_frames():
    fr = set()
    for k in (1, 2, 3):
        inp, tgt = window_frames(1, k)
        fr |= set(inp) | set(tgt[:2])
    return sorted(fr)


def pick_fig5_video(test_u8):
    """First test video whose digits are separate in every displayed true frame (any speed)."""
    vids = (test_u8.float() / 255).numpy()
    fr = displayed_frames()
    for i in range(len(vids)):
        if frames_separate(vids[i], fr):
            return i
    raise RuntimeError("no clean video found")


def figure5(model, model_label, test_u8, i, device, out):
    rows, stats = [], {}
    for k in (1, 2, 3):
        X, Y, mask = test_windows(test_u8[i:i + 1], k)
        P = predict(model, X, device)
        inp, tgt = window_frames(1, k)
        e2 = ((P[0, :2] - Y[0, :2]) ** 2).mean().item()
        stats[f"{k}x"] = e2
        rows.append((f"{k}x  E(2)={fmt(e2)}", X[0], P[0, :2], Y[0, :2], inp, tgt[:2]))
    fig = speed_panel(rows, f"{model_label}, test video {TEST_IDX.start + i} (test #{i}) at 1x, 2x, 3x")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return stats
