"""Phase 1 (before Q2.5): model A at 1x only. Table 2 and Figure 2.

Does NOT run any mirror, rotation or speed test.
"""
import json

import matplotlib.pyplot as plt

from pa2.data import default_device, ensure_dir, load_moving_mnist, split, n_blobs, TEST_IDX
from pa2.evaluate import (count_params, evaluate, fmt, fmt_seeds, get_predictor, predict, save_eval,
                          test_windows)
from pa2.plots import frame_grid, stack_pngs

device = default_device()
data = load_moving_mnist("data")
_, _, test = split(data)
ensure_dir("results/eval")
ensure_dir("figures")

names = ["A_s0", "A_s1", "Blank", "CopyLast"]
S = {}
for n in names:
    res = evaluate(get_predictor(n, device), test, device, parts=("1x",))
    S[n] = save_eval(n, res, {"params": count_params(n)})
    print(n, {k: round(v, 6) for k, v in S[n].items()})

# ---- Table 2
cols = [f"MSE{h}" for h in range(1, 6)] + ["E5_1x"]
lines = ["| Model | " + " | ".join(["MSE<sub>1</sub>", "MSE<sub>2</sub>", "MSE<sub>3</sub>",
                                      "MSE<sub>4</sub>", "MSE<sub>5</sub>", "E<sup>(5)</sup>"])
         + " | Trainable params |", "|---" * 8 + "|"]
lines.append("| A (baseline ConvLSTM) | " + " | ".join(fmt_seeds([S["A_s0"][c], S["A_s1"][c]]) for c in cols)
             + f" | {S['A_s0']['params']:,} |")
for n, lab in [("Blank", "Blank"), ("CopyLast", "Copy-last")]:
    lines.append(f"| {lab} | " + " | ".join(fmt(S[n][c]) for c in cols) + " | 0 |")
table2 = "\n".join(lines)
open("results/table2.md", "w").write(table2 + "\n")
print(table2)

# ---- Figure 2: two videos with separate digits in all 10 frames, one with overlap in the future
vids = (test.float() / 255).numpy()
sep, ovl = [], []
for i in range(len(vids)):
    nb = [n_blobs(vids[i, f, 0]) for f in range(10)]
    if min(nb) >= 2 and len(sep) < 2:
        sep.append(i)
    if min(nb[5:10]) == 1 and len(ovl) < 1:
        ovl.append(i)
    if len(sep) == 2 and ovl:
        break
chosen = sep + ovl
X1, Y1, _ = test_windows(test[chosen], 1)
P = predict(get_predictor("A_s0", device), X1, device)
paths = []
for j, i in enumerate(chosen):
    kind = "digits overlap" if i in ovl else "digits separate"
    mse = ((P[j] - Y1[j]) ** 2).mean().item()
    fig = frame_grid([list(X1[j]), list(P[j]), list(Y1[j])],
                     ["input (frames 1-5)", "prediction A (s0)", "ground truth (6-10)"],
                     ["1 | 6", "2 | 7", "3 | 8", "4 | 9", "5 | 10"],
                     f"test video {TEST_IDX.start + i} (test #{i}), {kind}, per-video E(5) = {fmt(mse)}",
                     cell=0.85, label_width=1.7)
    p = f"figures/fig2_v{i}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
stack_pngs(paths, "figures/fig2_baseline_examples.png")
json.dump({"fig2_test_positions": chosen, "fig2_dataset_indices": [TEST_IDX.start + i for i in chosen],
           "overlap": ovl}, open("results/fig2_videos.json", "w"), indent=1)
print("Figure 2 videos (test positions):", chosen)
