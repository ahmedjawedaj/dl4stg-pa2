"""Figure 1: Farneback flow direction and speed histograms on the first 50 training videos."""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pa2.data import load_moving_mnist, n_blobs, split, ensure_dir

data = load_moving_mnist("data")
train, _, _ = split(data)
vids = train[:50].numpy()[:, :, 0]  # (50, 20, 64, 64) uint8 = round(255 * x)

dxs, dys, ovl = [], [], []  # ovl: pixel from a pair where the two digits touch or overlap
n_pairs = 0
for v in vids:
    for s in range(19):
        prev, nxt = v[s], v[s + 1]
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        keep = (prev / 255.0 > 0.2) & (nxt / 255.0 > 0.2) & (np.sqrt(dx ** 2 + dy ** 2) > 0.5)
        dxs.append(dx[keep])
        dys.append(dy[keep])
        ovl.append(np.full(int(keep.sum()), min(n_blobs(prev / 255.0), n_blobs(nxt / 255.0)) < 2))
        n_pairs += 1
dx = np.concatenate(dxs)
dy = np.concatenate(dys)
ovl = np.concatenate(ovl)
theta = np.degrees(np.arctan2(-dy, dx)) % 360.0
speed = np.sqrt(dx ** 2 + dy ** 2)
n = len(speed)
med = float(np.median(speed))

dir_counts, edges = np.histogram(theta, bins=18, range=(0, 360))
dir_frac = dir_counts / n
stats = {
    "n_pairs": n_pairs, "n_pixels": int(n), "median_speed": med,
    "mean_speed": float(speed.mean()),
    "speed_percentiles": {str(q): float(np.percentile(speed, q)) for q in (10, 50, 90, 99)},
    "dir_fraction": dir_frac.tolist(), "dir_bin_edges": edges.tolist(),
    "dir_min_frac": float(dir_frac.min()), "dir_max_frac": float(dir_frac.max()),
    "speed_2x": 2 * med, "speed_3x": 3 * med,
    "frac_above_6px": float((speed > 6).mean()),
    "overlap_pixel_frac": float(ovl.mean()),
    "overlap_median_speed": float(np.median(speed[ovl])) if ovl.any() else None,
    "nonoverlap_median_speed": float(np.median(speed[~ovl])),
    "overlap_frac_below_p10": float(ovl[speed < np.percentile(speed, 10)].mean()),
}
# mirroring maps theta -> 180 - theta, rotating by 90 deg maps theta -> theta + 90
th_m = (180.0 - theta) % 360.0
th_r = (theta + 90.0) % 360.0
fm = np.histogram(th_m, bins=18, range=(0, 360))[0] / n
fr = np.histogram(th_r, bins=18, range=(0, 360))[0] / n
stats["mirror_l1"] = float(np.abs(dir_frac - fm).sum())
stats["rot90_l1"] = float(np.abs(dir_frac - fr).sum())
stats["uniform_l1"] = float(np.abs(dir_frac - 1 / 18).sum())
# does overlap bias the histograms? compare pairs with touching/overlapping digits to separate ones
d_sep = np.histogram(theta[~ovl], bins=18, range=(0, 360))[0] / (~ovl).sum()
stats["dir_sep_min_frac"] = float(d_sep.min())
stats["dir_sep_max_frac"] = float(d_sep.max())
stats["dir_sep_vs_all_l1"] = float(np.abs(d_sep - dir_frac).sum())

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(edges[:-1], dir_frac, width=20, align="edge", color="#4C72B0", edgecolor="white")
ax[0].axhline(1 / 18, color="k", ls="--", lw=1.2, label="uniform (1/18)")
ax[0].set_xlim(0, 360)
ax[0].set_xticks(range(0, 361, 45))
ax[0].set_xlabel(r"flow direction $\theta$ = atan2($-d_y$, $d_x$) (degrees, 0 = right, 90 = up)")
ax[0].set_ylabel("fraction of kept pixels")
ax[0].set_title("(a) flow direction")
ax[0].legend(loc="lower right")
bins = np.arange(0.5, 6.01, 0.25)  # x-axis cut at 6 px, fraction above is in the stats
w = np.ones_like(speed) / n
ax[1].hist(speed, bins=bins, weights=w, color="#DD8452", edgecolor="white")
ax[1].axvline(med, color="k", lw=1.5, label=f"median = {med:.2f} px/frame")
ax[1].set_xlabel("flow speed (pixels per frame)")
ax[1].set_ylabel("fraction of kept pixels")
ax[1].set_title("(b) flow speed")
ax[1].legend()
fig.tight_layout()
ensure_dir("figures")
fig.savefig("figures/fig1_flow.png", dpi=160)
json.dump(stats, open("results/fig1_flow_stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in stats.items() if k not in ("dir_bin_edges",)}, indent=1))
