"""Phase 3 (only after Q2.11 is committed and B-D are trained): Tables 5 and 6, Figure 6.

Usage: python -m scripts.phase3_final            (A-D, symmetrised B and D, final = symmetrised D)
       python -m scripts.phase3_final --db       (adds D+B rows, final = symmetrised D+B)
D+B is optional: train it only if Table 5 supports it, and say why in the report.
"""
import argparse
import json

from pa2.data import default_device, ensure_dir, load_moving_mnist, split
from pa2.evaluate import count_params, evaluate, fmt_seeds, get_predictor, inference_time, save_eval
from pa2.figs import figure3, figure5

ap = argparse.ArgumentParser()
ap.add_argument("--db", action="store_true", help="also evaluate D+B and use symmetrised D+B as final")
ap.add_argument("--final", default=None, choices=["D", "DB"], help="final model (default D, or DB with --db)")
args = ap.parse_args()

device = default_device()
data = load_moving_mnist("data")
_, _, test = split(data)
ensure_dir("results/eval")
d_arch = json.load(open("results/d_selection.json"))["chosen"]

learned = {"A": ["A_s0", "A_s1"], "B": ["B_s0", "B_s1"], "C": ["C_s0", "C_s1"],
           "D": [f"D_{d_arch}_s0", f"D_{d_arch}_s1"]}
if args.db:
    learned["DB"] = [f"DB_{d_arch}_s0", f"DB_{d_arch}_s1"]
sym = {"sym B": [f"sym:{t}" for t in learned["B"]], "sym D": [f"sym:{t}" for t in learned["D"]]}
if args.db:
    sym["sym D+B"] = [f"sym:{t}" for t in learned["DB"]]

S = {}
for group in list(learned.values()) + list(sym.values()):
    for name in group:
        pred = get_predictor(name, device)
        res = evaluate(pred, test, device, parts=("1x", "M", "R", "speed"))
        base = name.replace("sym:", "")
        log = json.load(open(f"results/train_logs/{base}.json"))
        extra = {"params": count_params(name), "train_time_s": log["mean_epoch_time_s"],
                 "infer_time_s": inference_time(pred, test, device), "device": log["device"]}
        S[name] = save_eval(name, res, extra)
        print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in S[name].items()},
              flush=True)

keys = ["E5_1x", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x"]
head = ("| Model | E<sup>(5)</sup> 1x | C<sub>M</sub> | E<sub>M</sub> | C<sub>R</sub> | E<sub>R</sub> | "
        "E<sup>(2)</sup> 1x | E<sup>(2)</sup> 2x | E<sup>(2)</sup> 3x | Params | Train s/epoch | Inference s |")


def table(groups, labels, sci_gaps=False):
    """sci_gaps: Table 6 gives C_M and C_R of the wrapped models in scientific notation."""
    lines = [head, "|---" * 12 + "|"]
    for g, lab in zip(groups, labels):
        cells = [fmt_seeds([S[n][k] for n in g], sci=sci_gaps and k.startswith("C_")) for k in keys]
        cells.append(f"{S[g[0]]['params']:,}")
        cells.append(fmt_seeds([S[n]["train_time_s"] for n in g]))
        cells.append(fmt_seeds([S[n]["infer_time_s"] for n in g]))
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


labs5 = {"A": "A (baseline, 1x)", "B": "B (baseline, 1x, sym. aug.)", "C": "C (baseline, 1x/2x/3x)",
         "D": f"D ({d_arch}, 1x/2x/3x)", "DB": f"D+B ({d_arch}, 1x/2x/3x, sym. aug.)"}
order5 = [k for k in ("A", "B", "C", "D", "DB") if k in learned]
t5 = table([learned[k] for k in order5], [labs5[k] for k in order5])
open("results/table5.md", "w").write(t5 + "\n")
labs6 = {"sym B": "symmetrised B", "sym D": f"symmetrised D ({d_arch})",
         "sym D+B": f"symmetrised D+B ({d_arch})"}
final = args.final or ("DB" if args.db else "D")
final_key = "sym D+B" if final == "DB" else "sym D"
t6 = table(list(sym.values()), [labs6[k] + (" = final model" if k == final_key else "") for k in sym],
           sci_gaps=True)
open("results/table6.md", "w").write(t6 + "\n")
print(t5)
print(t6)

# ---- Figure 6: final model on exactly the Figure 3 and Figure 5 videos (seed 0)
final_name = f"sym:{learned[final][0]}"
model = get_predictor(final_name, device)
f3 = json.load(open("results/fig3_videos.json"))
picks = [(p["T"], p["test_pos"], p["C"]) for p in f3["picks"]]
label = f"final model: symmetrised {'D+B' if final == 'DB' else 'D'} ({d_arch}, seed 0)"
png, gifs, stats = figure3(model, label, test, picks, device, "fig6a_final")
i5 = json.load(open("results/fig5_video.json"))["test_pos"]
st5 = figure5(model, label, test, i5, device, "figures/fig6b_final_speeds.png")
json.dump({"final": final_name, "fig6a_gifs": gifs, "fig6a_stats": stats, "fig6b_E2": st5,
           "d_arch": d_arch}, open("results/fig6.json", "w"), indent=1)
