"""Moving MNIST loading, video-level splits, frame windows and square symmetries.

Every experiment in this repository calls these functions, including the
Table 1 checks in scripts/table1_checks.py.
"""
from __future__ import annotations

import os

import numpy as np
import torch

N_FRAMES = 20
N_IN = 5
N_OUT = 5

TRAIN_IDX = range(0, 8000)
VAL_IDX = range(8000, 9000)
TEST_IDX = range(9000, 10000)

# Allowed start frames t (1-based) for each speed k.
# Normal-speed training (A, B): all five targets exist -> t in 1..11 at k=1.
NORMAL_STARTS = {1: list(range(1, 12))}
# Speed-augmented training (C, D): all five inputs and >= 2 targets exist.
AUG_STARTS = {1: list(range(1, 15)), 2: list(range(1, 9)), 3: [1, 2]}


def load_moving_mnist(root: str = "data") -> torch.Tensor:
    """Return all 10,000 videos as uint8 tensor (10000, 20, 1, 64, 64).

    Uses torchvision with split=None, so every video keeps all 20 frames.
    """
    import torchvision

    ds = torchvision.datasets.MovingMNIST(root=root, split=None, download=True)
    data = ds.data if hasattr(ds, "data") else torch.stack([ds[i] for i in range(len(ds))])
    assert data.shape == (10000, 20, 1, 64, 64), data.shape
    return data


def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 in [0, 1]."""
    return x.float().div_(255.0)


def split(data: torch.Tensor):
    """Manual split by dataset index (whole videos)."""
    if os.environ.get("PA2_SMOKE"):  # code smoke tests only: tiny subsets
        n = int(os.environ["PA2_SMOKE"])
        print(f"WARNING: PA2_SMOKE={n}, using tiny subsets")
        return data[:n], data[8000:8000 + n], data[9000:9000 + n]
    return (data[TRAIN_IDX.start:TRAIN_IDX.stop],
            data[VAL_IDX.start:VAL_IDX.stop],
            data[TEST_IDX.start:TEST_IDX.stop])


# ---------------------------------------------------------------- windows
def window_frames(t: int, k: int):
    """1-based frame numbers of the input and of the existing target frames."""
    inp = [t + k * i for i in range(N_IN)]
    tgt = [t + k * (N_IN + i) for i in range(N_OUT)]
    tgt = [f for f in tgt if f <= N_FRAMES]
    return inp, tgt


def get_window(video: torch.Tensor, t: int, k: int = 1):
    """Window of a video (..., 20, C, H, W) starting at 1-based frame t at speed k.

    Returns X (..., 5, C, H, W), Y (..., 5, C, H, W) and a bool mask (5,).
    Missing target frames are zero-filled in Y and False in the mask, so
    they never enter a loss or a metric.
    """
    inp, tgt = window_frames(t, k)
    assert max(inp) <= N_FRAMES, "all five input frames must exist"
    X = video[..., [f - 1 for f in inp], :, :, :]
    Y = torch.zeros_like(X)
    if tgt:
        Y[..., : len(tgt), :, :, :] = video[..., [f - 1 for f in tgt], :, :, :]
    mask = torch.zeros(N_OUT, dtype=torch.bool)
    mask[: len(tgt)] = True
    return X, Y, mask


def batch_window_indices(t: torch.Tensor, k: torch.Tensor):
    """Vectorised window indices for a batch of (t, k) pairs.

    Returns 0-based frame indices (B, 10) (invalid targets clamped to 0)
    and a bool target mask (B, 5).
    """
    steps = torch.arange(N_IN + N_OUT, device=t.device)
    frames = t[:, None] + k[:, None] * steps[None, :]  # 1-based
    mask = frames[:, N_IN:] <= N_FRAMES
    idx = torch.where(frames <= N_FRAMES, frames - 1, torch.zeros_like(frames))
    return idx, mask


# ---------------------------------------------------------------- symmetries
def mirror(x: torch.Tensor) -> torch.Tensor:
    """(Mx)[i, j] = x[i, 63 - j], applied to the last two (row, column) axes."""
    return torch.flip(x, dims=(-1,))


def rotate(x: torch.Tensor, a: int = 1) -> torch.Tensor:
    """R^a: a quarter turns anticlockwise."""
    return torch.rot90(x, k=a % 4, dims=(-2, -1))


# Group G = {R^a, M R^a : a = 0..3}. Element index g = 4*m + a.
G = [(m, a) for m in (0, 1) for a in range(4)]


def apply_g(x: torch.Tensor, g: int) -> torch.Tensor:
    """(M R^a)(x) = M(R^a(x)): rotate first, then mirror."""
    m, a = G[g]
    y = rotate(x, a)
    return mirror(y) if m else y


def apply_g_inv(x: torch.Tensor, g: int) -> torch.Tensor:
    """Inverse: R^a -> R^-a, and every M R^a is its own inverse."""
    m, a = G[g]
    if m:
        return apply_g(x, g)
    return rotate(x, -a)


# ---------------------------------------------------------------- overlap heuristic
def n_blobs(frame: np.ndarray, thr: float = 0.2) -> int:
    """Number of 8-connected bright blobs (used only to pick figure videos)."""
    import cv2

    mask = (frame > thr).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # ignore background and specks
    return int(sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 15))


def frames_separate(video: np.ndarray, frame_numbers) -> bool:
    """True if every listed 1-based frame shows two separate blobs."""
    return all(n_blobs(video[f - 1, 0]) >= 2 for f in frame_numbers)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p
