"""ConvLSTM forecasters (model A baseline and model D candidates) and reference predictors."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import apply_g, apply_g_inv, N_IN, N_OUT

N_STEPS = 9  # 5 observed inputs + 4 fed-back predictions


class ConvLSTMCell(nn.Module):
    """Shi et al. (2015) ConvLSTM without peepholes.

    [i, f, o, g] = Conv([u_t, h_{t-1}])
    c_t = sigma(f) * c_{t-1} + sigma(i) * tanh(g)
    h_t = sigma(o) * tanh(c_t)
    """

    def __init__(self, in_ch: int, hid: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.hid = hid
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, kernel, stride=1, padding=pad,
                              dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


@dataclass
class NetConfig:
    patch: int = 4                      # pixel_unshuffle factor
    hidden: List[int] = field(default_factory=lambda: [64, 64])
    kernels: List[int] = field(default_factory=lambda: [3, 3])
    dilations: List[int] = field(default_factory=lambda: [1, 1])
    name: str = "A"


class ConvLSTMForecaster(nn.Module):
    """Patch-fold -> stacked ConvLSTM -> 1x1 conv -> pixel_shuffle -> sigmoid.

    With the default config this is exactly model A (480,784 parameters).
    """

    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch
        in_ch = p * p
        cells = []
        for hid, k, d in zip(cfg.hidden, cfg.kernels, cfg.dilations):
            cells.append(ConvLSTMCell(in_ch, hid, k, d))
            in_ch = hid
        self.cells = nn.ModuleList(cells)
        self.out = nn.Conv2d(cfg.hidden[-1], p * p, 1, bias=True)

    def step(self, frame, states):
        u = F.pixel_unshuffle(frame, self.cfg.patch)
        new_states = []
        for cell, st in zip(self.cells, states):
            h, c = cell(u, st)
            new_states.append((h, c))
            u = h
        x_next = torch.sigmoid(F.pixel_shuffle(self.out(u), self.cfg.patch))
        return x_next, new_states

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, 5, 1, 64, 64) -> Y_hat: (B, 5, 1, 64, 64).

        9-step rollout without teacher forcing: steps 1-5 read the input
        frames, steps 6-9 read the model's own previous prediction.
        Outputs of steps 5-9 form Y_hat. Outputs of steps 1-4 are dropped.
        Gradients flow through the fed-back predictions.
        """
        B, _, _, H, W = X.shape
        p = self.cfg.patch
        states = [(X.new_zeros(B, c.hid, H // p, W // p), X.new_zeros(B, c.hid, H // p, W // p))
                  for c in self.cells]
        outs = []
        pred = None
        for tau in range(1, N_STEPS + 1):
            u = X[:, tau - 1] if tau <= N_IN else pred
            pred, states = self.step(u, states)
            if tau >= N_IN:
                outs.append(pred)
        return torch.stack(outs, dim=1)

    def reach_pixels(self) -> int:
        """One-step reach (original pixels) under the handout's conventions.

        A conv with kernel k and dilation d moves information d*(k-1)/2 grid
        cells. The longest path inside one step starts at u_tau (or h^(1)_{tau-1})
        and passes through every layer's conv once. The 1x1 output conv and the
        (un)shuffles move nothing. One cell is `patch` original pixels.
        """
        cells = sum(d * (k - 1) // 2 for k, d in zip(self.cfg.kernels, self.cfg.dilations))
        return cells * self.cfg.patch


class Blank(nn.Module):
    def forward(self, X):
        return torch.zeros_like(X[:, :N_OUT])


class CopyLast(nn.Module):
    def forward(self, X):
        return X[:, -1:].expand(-1, N_OUT, -1, -1, -1).clone()


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^{-1}( f(g(X)) ) over the 8 square symmetries."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, X):
        acc = None
        for g in range(8):
            y = apply_g_inv(self.base(apply_g(X, g)), g)
            acc = y if acc is None else acc + y
        return acc / 8.0


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ registry
CONFIGS = {
    "A": NetConfig(name="A"),
    # model D candidates (all <= 721,176 parameters, reach > 8 px, no attention)
    # 5x5 kernels in both layers: 2 + 2 cells = 16 px
    "D_k5": NetConfig(patch=4, hidden=[46, 46], kernels=[5, 5], dilations=[1, 1], name="D_k5"),
    # 3x3 kernels, layer 2 dilated by 2: 1 + 2 cells = 12 px
    "D_dil": NetConfig(patch=4, hidden=[78, 78], kernels=[3, 3], dilations=[1, 2], name="D_dil"),
    # Additional validation experiment: keep C's width and change only layer-2 dilation.
    "D_dil64": NetConfig(patch=4, hidden=[64, 64], kernels=[3, 3], dilations=[1, 2], name="D_dil64"),
    # 8x8 patch fold, 3x3 kernels: 2 cells of 8 px = 16 px
    "D_p8": NetConfig(patch=8, hidden=[70, 70], kernels=[3, 3], dilations=[1, 1], name="D_p8"),
}


def build(arch: str) -> ConvLSTMForecaster:
    return ConvLSTMForecaster(CONFIGS[arch])


def config_dict(arch: str):
    return asdict(CONFIGS[arch])
