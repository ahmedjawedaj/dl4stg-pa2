"""Table 1: transformation and frame-window checks, using the experiment functions."""
import json

import torch

from pa2.data import get_window, mirror, rotate, ensure_dir

torch.manual_seed(0)
x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64)
p[7, 0, 10, 20] = 1


def bright(v):
    idx = (v == v.max()).nonzero()
    assert len(idx) == 1
    f, _, r, c = idx[0].tolist()
    return (f, r, c)


rows = []
for name, lhs, rhs, expected in [
    ("(a) max|M(M(x)) - x|", mirror(mirror(x)), x, "0"),
    ("(b) max|R^4(x) - x|", rotate(rotate(rotate(rotate(x)))), x, "0"),
    ("(c) max|M R M(x) - R^-1(x)|", mirror(rotate(mirror(x))), rotate(x, -1), "0"),
]:
    d = (lhs - rhs).abs().max().item()
    rows.append({"check": name, "expected": expected,
                 "yours": f"{d:g} (torch.equal={torch.equal(lhs, rhs)})"})

rows.append({"check": "(d) bright pixel of M(p) (frame, row, column)", "expected": "(7, 10, 43)",
             "yours": str(bright(mirror(p)))})
rows.append({"check": "(e) bright pixel of R(p) (frame, row, column)", "expected": "(7, 43, 10)",
             "yours": str(bright(rotate(p)))})

v = torch.arange(1, 21).float().view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
for lab, k, expected in [("(f) 1x", 1, "1-5 / 6-10"),
                         ("(g) 2x", 2, "1, 3, 5, 7, 9 / 11, 13, 15, 17, 19"),
                         ("(h) 3x", 3, "1, 4, 7, 10, 13 / 16, 19")]:
    X, Y, mask = get_window(v, 1, k)
    xin = [int(X[i, 0, 0, 0]) for i in range(5)]
    yt = [int(Y[i, 0, 0, 0]) for i in range(5) if mask[i]]
    rows.append({"check": f"{lab} input / target frames", "expected": expected,
                 "yours": ", ".join(map(str, xin)) + " / " + ", ".join(map(str, yt))})

for r in rows:
    print(f"{r['check']:<50} {r['expected']:<38} {r['yours']}")
ensure_dir("results")
json.dump(rows, open("results/table1.json", "w"), indent=1)
