"""Figure and GIF helpers."""
from __future__ import annotations

import io

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _img(ax, frame, title=None):
    ax.imshow(np.asarray(frame).squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def frame_grid(rows, row_labels, col_titles=None, suptitle=None, cell=0.62, sep_after=None,
               label_width=1.6):
    """rows: list of lists of (64,64) frames. Draws a grid, with an optional gap after a column."""
    nr, nc = len(rows), max(len(r) for r in rows)
    fig = plt.figure(figsize=(label_width + cell * nc + (0.15 if sep_after else 0), cell * nr + 0.45))
    left = label_width / fig.get_figwidth()
    gs = fig.add_gridspec(nr, nc, left=left, right=0.995, top=1 - 0.4 / fig.get_figheight(),
                          bottom=0.02, wspace=0.05, hspace=0.08)
    for i, r in enumerate(rows):
        for j, fr in enumerate(r):
            ax = fig.add_subplot(gs[i, j])
            if sep_after is not None and j >= sep_after:
                pos = ax.get_position()
                ax.set_position([pos.x0 + 0.012, pos.y0, pos.width, pos.height])
            _img(ax, fr, col_titles[j] if (col_titles and i == 0) else None)
            if j == 0:
                ax.text(-0.08, 0.5, row_labels[i], transform=ax.transAxes, ha="right", va="center",
                        fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=0.995)
    return fig


def four_row_panel(X, fX, TX, TY, fTX, TfX, tname, title):
    """Figure 3 layout: rows (1) X|f(X) (2) T(X)|T(Y) (3) T(X)|f(T(X)) (4) T(X)|T(f(X))."""
    T = tname
    rows = [list(X) + list(fX), list(TX) + list(TY), list(TX) + list(fTX), list(TX) + list(TfX)]
    labels = ["(1) X | f(X)", f"(2) {T}(X) | {T}(Y)", f"(3) {T}(X) | f({T}(X))",
              f"(4) {T}(X) | {T}(f(X))"]
    cols = [f"in {i}" for i in range(1, 6)] + [f"h={h}" for h in range(1, 6)]
    return frame_grid(rows, labels, cols, title, sep_after=5)


def four_row_gif(path, X, fX, TX, TY, fTX, TfX, tname, title, scale=1):
    """GIF stepping through the 5 horizons, keeping the same four rows (5 inputs + horizon h)."""
    frames = []
    T = tname
    labels = ["(1) X | f(X)", f"(2) {T}(X) | {T}(Y)", f"(3) {T}(X) | f({T}(X))",
              f"(4) {T}(X) | {T}(f(X))"]
    for h in range(5):
        rows = [list(X) + [fX[h]], list(TX) + [TY[h]], list(TX) + [fTX[h]], list(TX) + [TfX[h]]]
        cols = [f"in {i}" for i in range(1, 6)] + [f"horizon h={h + 1}"]
        fig = frame_grid(rows, labels, cols, title, cell=0.9, sep_after=5, label_width=1.7)
        frames.append(fig_to_array(fig))
        plt.close(fig)
    imageio.mimsave(path, frames, duration=900, loop=0)


def fig_to_array(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return imageio.imread(buf)


def stack_pngs(paths, out, gap=12):
    """Vertically stack PNG panels into one figure file."""
    ims = [imageio.imread(p) for p in paths]
    w = max(i.shape[1] for i in ims)
    padded = []
    for im in ims:
        if im.shape[2] == 4:
            im = im[..., :3]
        pad = np.full((im.shape[0], w - im.shape[1], 3), 255, np.uint8)
        padded.append(np.concatenate([im, pad], 1))
        padded.append(np.full((gap, w, 3), 255, np.uint8))
    imageio.imwrite(out, np.concatenate(padded[:-1], 0))


def speed_panel(rows_by_speed, title):
    """Figure 5 layout. rows_by_speed: list of (label, inputs(5), preds(2), trues(2), in_frames, tgt_frames)."""
    rows, labels = [], []
    for lab, xin, pr, tr, fin, ftg in rows_by_speed:
        rows.append(list(xin) + list(pr) + list(tr))
        labels.append(f"{lab}\nin {','.join(map(str, fin))}\ntarget {','.join(map(str, ftg))}")
    cols = [f"in {i}" for i in range(1, 6)] + ["pred h=1", "pred h=2", "true h=1", "true h=2"]
    return frame_grid(rows, labels, cols, title, cell=0.8, label_width=1.9)
