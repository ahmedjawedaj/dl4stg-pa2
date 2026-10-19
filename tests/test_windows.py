"""Sanity checks: batched training windows match get_window, start rules, augmentation."""
import torch
from pa2.data import batch_window_indices, get_window, apply_g, apply_g_inv, AUG_STARTS
from pa2.train import sample_windows

v = torch.arange(1, 21).float().view(20, 1, 1, 1).expand(20, 1, 64, 64)  # frame f filled with f
gen = torch.Generator().manual_seed(0)
for regime in ("normal", "speed"):
    t, k = sample_windows(20000, regime, gen)
    for kk in (1, 2, 3):
        ts = t[k == kk]
        if len(ts):
            print(regime, kk, "count", len(ts), "t range", ts.min().item(), ts.max().item(), "distinct", ts.unique().numel())
    idx, mask = batch_window_indices(t[:500], k[:500])
    for i in range(500):
        X, Y, m = get_window(v, int(t[i]), int(k[i]))
        Xb = v[idx[i, :5]]
        Yb = v[idx[i, 5:]] * mask[i][:, None, None, None]
        assert torch.equal(X, Xb) and torch.equal(Y, Yb) and torch.equal(m, mask[i])
        assert m.sum() >= 2 and (regime == "speed" or m.all())
print("batched windows match get_window")
x = torch.rand(3, 5, 1, 64, 64)
for g in range(8):
    assert torch.equal(apply_g_inv(apply_g(x, g), g), x)
print("all 8 inverses exact, 8 distinct elements:",
      len({apply_g(x, g)[0,0,0].flatten()[:50].numpy().tobytes() for g in range(8)}) == 8)
