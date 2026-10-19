"""Phase 2 (only after Q2.5 is committed): stress tests of model A.
Tables 3 and 4, Figures 3, 4 and 5 (+ Figure 3 GIFs)."""
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from pa2.data import TEST_IDX, default_device, ensure_dir, load_moving_mnist, split
from pa2.evaluate import count_params, evaluate, fmt, fmt_seeds, get_predictor, save_eval
from pa2.figs import figure3, figure5, pick_fig5_video

device = default_device()
data = load_moving_mnist("data")
_, _, test = split(data)
for d in ("results/eval", "figures", "gifs"):
    ensure_dir(d)

names = ["A_s0", "A_s1", "Blank", "CopyLast"]
S, R = {}, {}
for n in names:
    R[n] = evaluate(get_predictor(n, device), test, device, parts=("1x", "M", "R", "speed"))
    S[n] = save_eval(n, R[n], {"params": count_params(n)})
    print(n, {k: round(v, 6) for k, v in S[n].items()})


def row(label, key_fn, keys):
    return f"| {label} | " + " | ".join(key_fn(k) for k in keys) + " |"


A = lambda k: fmt_seeds([S["A_s0"][k], S["A_s1"][k]])
# ---- Table 3
k3 = ["E5_1x", "C_M", "E_M", "C_R", "E_R"]
t3 = ["| Model | E<sup>(5)</sup> | C<sub>M</sub> | E<sub>M</sub> | C<sub>R</sub> | E<sub>R</sub> |",
      "|---|---|---|---|---|---|", row("A", A, k3),
      row("Blank", lambda k: fmt(S["Blank"][k]), k3),
      row("Copy-last", lambda k: fmt(S["CopyLast"][k]), k3)]
open("results/table3.md", "w").write("\n".join(t3) + "\n")
# ---- Table 4
k4 = ["E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
t4 = ["| Model | E<sup>(2)</sup> 1x | E<sup>(2)</sup> 2x | E<sup>(2)</sup> 3x | E<sup>(5)</sup> 1x | E<sup>(5)</sup> 2x |",
      "|---|---|---|---|---|---|", row("A", A, k4),
      row("Copy-last", lambda k: fmt(S["CopyLast"][k]), k4)]
open("results/table4.md", "w").write("\n".join(t4) + "\n")
print("\n".join(t3))
print("\n".join(t4))

# ---- Figure 3: two largest per-video C_M and two largest per-video C_R (seed-0 A)
r0 = R["A_s0"]
picks = []
for T in ("M", "R"):
    order = np.argsort(-r0[f"C_{T}"])[:2]
    picks += [(T, int(i), float(r0[f"C_{T}"][i])) for i in order]
model = get_predictor("A_s0", device)
png, gifs, stats3 = figure3(model, "model A (seed 0)", test, picks, device, "fig3_A")
json.dump({"picks": [{"T": T, "test_pos": i, "dataset_idx": TEST_IDX.start + i, "C": c}
                     for T, i, c in picks], "gifs": gifs, "panel_stats": stats3},
          open("results/fig3_videos.json", "w"), indent=1)

# ---- Figure 4: histograms of per-video C_M and C_R (seed-0 A), Figure 3 videos marked
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, T, col in [(ax[0], "M", "#4C72B0"), (ax[1], "R", "#C44E52")]:
    c = r0[f"C_{T}"]
    bins = np.logspace(np.log10(c.min() * 0.9), np.log10(c.max() * 1.1), 50)
    a.hist(c, bins=bins, color=col, alpha=0.85, edgecolor="white")
    a.set_xscale("log")
    ticks = [0.0005, 0.001, 0.002, 0.005] if T == "M" else [0.001, 0.002, 0.004, 0.006]
    a.xaxis.set_major_locator(FixedLocator(ticks))
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.axvline(c.mean(), color="k", ls="--", lw=1, label=f"mean = {fmt(float(c.mean()))}")
    a.axvline(np.median(c), color="gray", ls=":", lw=1.2, label=f"median = {fmt(float(np.median(c)))}")
    for T2, i, v in picks:
        if T2 == T:
            a.axvline(v, color="orange", lw=1.6)
            a.annotate(f"#{i}", (v, a.get_ylim()[1] * 0.9), rotation=90, fontsize=8, ha="right",
                       va="top", color="darkorange")
    a.set_xlabel(f"per-video consistency gap C_{T} (log scale)")
    a.set_ylabel("number of test videos")
    a.set_title(f"({'a' if T == 'M' else 'b'}) {'mirror' if T == 'M' else 'rotation'}, model A seed 0")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig4_consistency_hist.png", dpi=160)
plt.close(fig)

# ---- Figure 5: one video with separate digits in all displayed true frames, 1x/2x/3x
i5 = pick_fig5_video(test)
st5 = figure5(model, "model A (seed 0)", test, i5, device, "figures/fig5_A_speeds.png")
json.dump({"test_pos": i5, "dataset_idx": TEST_IDX.start + i5, "per_video_E2": st5},
          open("results/fig5_video.json", "w"), indent=1)
print("Figure 3 picks", picks, "Figure 5 video", i5)
