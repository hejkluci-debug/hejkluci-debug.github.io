import json, sys, itertools
import numpy as np
import qart_layout as L
URL = "HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/"
res = []
for v, e in ((9, "M"), (10, "M"), (11, "M"), (9, "Q"), (11, "Q")):
    n = 4 * v + 17
    cands = []
    for sym in (1400, 1550, 1700):
        m = sym / n
        for y0 in range(160, 700, 60):
            for x0 in range(80, 500, 60):
                if not L.finders_free(x0, y0, m, n):
                    continue
                cov = float((L.cat_a[y0:y0 + sym, x0:x0 + sym] > 0.5).mean())
                cands.append((cov, x0, y0, sym))
    # bucket by coverage, try the best few per bucket
    for lo in (0.20, 0.30, 0.40, 0.50):
        bucket = sorted([c for c in cands if lo <= c[0] < lo + 0.1], key=lambda c: -c[3])[:3]
        for cov, x0, y0, sym in bucket:
            for mk in (0, 2, 5):
                r = L.run(URL, v, e, mk, x0, y0, sym, dots=False)
                if r is None:
                    continue
                row = {k: r[k] for k in ("version", "ecl", "mask", "x0", "y0", "sym_px", "cover", "score", "quick_ok", "zxing_unused_ec", "module_mm")}
                res.append(row)
                print(row, flush=True)
json.dump(res, open(L.OUT / "sweep_nodots.json", "w"), indent=1)
