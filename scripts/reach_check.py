"""One-step reach: analytic value (handout conventions) plus an autograd check of how many
grid cells each source of one step (u_tau, h^(l)_{tau-1}) can influence in x_hat_{tau+1}.

The autograd support also includes the within-patch spread of the 4x4 (or 8x8) fold. The
handout's convention counts centre-to-centre cell distance only, so we report the cell radius.
Usage: python -m scripts.reach_check A D_k5 D_dil D_p8
"""
import json
import os
import sys

import torch

from pa2.models import build, n_params

archs = sys.argv[1:] or ["A"]
out = json.load(open("results/reach.json")) if os.path.exists("results/reach.json") else {}
for arch in archs:
    torch.manual_seed(0)
    m = build(arch).double().eval()
    p = m.cfg.patch
    G = 64 // p
    c = G // 2
    srcs = {}
    states = [(torch.randn(1, cell.hid, G, G, dtype=torch.double),
               torch.randn(1, cell.hid, G, G, dtype=torch.double)) for cell in m.cells]
    frame = torch.rand(1, 1, 64, 64, dtype=torch.double)
    # source: input frame u_tau
    frame.requires_grad_(True)
    for st in states:
        st[0].requires_grad_(True)
    x_next, _ = m.step(frame, states)
    # output cell at grid centre (c, c): sum the p x p pixels of that cell
    tgt = x_next[0, 0, c * p:(c + 1) * p, c * p:(c + 1) * p].sum()
    grads = torch.autograd.grad(tgt, [frame] + [st[0] for st in states])
    g_in = grads[0][0, 0].abs().reshape(G, p, G, p).sum((1, 3))
    radius = {}
    for name, g in [("u_tau", g_in)] + [(f"h{l + 1}_prev", grads[l + 1][0].abs().sum(0))
                                         for l in range(len(states))]:
        nz = (g > 0).nonzero()
        r = int((nz - torch.tensor([c, c])).abs().max()) if len(nz) else 0
        radius[name] = r
    cells = max(radius.values())
    out[arch] = {"analytic_reach_px": m.reach_pixels(), "patch": p,
                 "autograd_cell_radius": radius, "autograd_reach_px": cells * p,
                 "params": n_params(build(arch)), "config": vars(m.cfg)}
    print(arch, out[arch])
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/reach.json", "w"), indent=1)
