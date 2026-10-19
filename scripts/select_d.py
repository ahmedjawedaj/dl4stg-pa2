"""Pick model D's architecture from validation results of the candidate runs (never the test split).

Selection rule, fixed before any candidate was trained: lowest mean validation E(2) over 1x, 2x
and 3x (t = 1), averaged over seeds 0 and 1, since D exists to handle all three training speeds.
"""
import json
from collections import defaultdict

from pa2.evaluate import fmt_seeds
from pa2.models import CONFIGS, build

runs = defaultdict(dict)
for arch in [a for a in CONFIGS if a.startswith("D_")]:
    for s in (0, 1):
        L = json.load(open(f"results/train_logs/D_{arch}_s{s}.json"))
        v = L["val_final"]
        v["score"] = (v["E2_1x"] + v["E2_2x"] + v["E2_3x"]) / 3
        v["params"], v["epochs"], v["epoch_time_s"] = L["params"], len(L["epochs"]), L["mean_epoch_time_s"]
        runs[arch][s] = v

rank = sorted(runs, key=lambda a: (runs[a][0]["score"] + runs[a][1]["score"]) / 2)
keys = ["E5_1x", "E2_1x", "E2_2x", "E2_3x", "score"]
lines = ["| Candidate | Reach (px) | Params | val E<sup>(5)</sup> 1x | val E<sup>(2)</sup> 1x | "
         "val E<sup>(2)</sup> 2x | val E<sup>(2)</sup> 3x | mean val E<sup>(2)</sup> (selection score) |",
         "|---" * 8 + "|"]
for a in rank:
    cells = [fmt_seeds([runs[a][0][k], runs[a][1][k]]) for k in keys]
    lines.append(f"| {a}{' (chosen)' if a == rank[0] else ''} | {build(a).reach_pixels()} | "
                 f"{runs[a][0]['params']:,} | " + " | ".join(cells) + " |")
open("results/d_selection.md", "w").write("\n".join(lines) + "\n")
json.dump({"chosen": rank[0], "runs": runs}, open("results/d_selection.json", "w"), indent=1)
print("\n".join(lines))
print("chosen:", rank[0])
