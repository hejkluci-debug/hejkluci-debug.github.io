"""Search QR version / EC level / mask / placement for the reference-style
layout, maximising how much of the QR the cat covers while staying readable.

Stage 1 (fast, exact): for every RS block count the codewords that MUST be
wrong - fixed data (URL, headers) under the cat whose fixed colour differs
from what the cat shows there. Reject configs where this already eats more
than `max_fixed_use` of the block's correction capacity.
Stage 2: full QArt solve + 3 decoders x 7 stress tests for the survivors.
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qr-engine"))
from qart import QArtEncoder, codeword_map
import qart_layout as L

URL = sys.argv[1] if len(sys.argv) > 1 else "HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/"
VERSIONS = [8, 9, 10, 11, 12, 13]
ECLS = ["L", "M", "Q"]
MAX_FIXED_USE = 0.5


def stage1(enc, cwmap, mask, x0, y0, sym):
    cw, block_of, is_data, data_index, dlens, eclen = cwmap
    n = enc.size; m = sym / n
    if not L.finders_free(x0, y0, m, n):
        return None
    cov, Lum = L.module_stats(x0, y0, m, n)
    covered = cov > 0.15
    tgt = (Lum < 150).astype(np.int8)
    base = enc.base_for_mask(mask)
    free_data_cw = set(np.array(enc.free) // 8)
    t = eclen // 2
    bad = np.zeros(len(dlens), int)
    for c in range(cw.max() + 1):
        if not is_data[c] or data_index[c] in free_data_cw:
            continue
        sel = (cw == c) & covered
        if sel.any() and (base[sel] != tgt[sel]).any():
            bad[block_of[c]] += 1
    cover = float((L.cat_a[int(y0):int(y0 + sym), int(x0):int(x0 + sym)] > 0.5).mean())
    return {"cover": cover, "fixed_bad": bad.tolist(), "t": t, "use": float(bad.max() / t)}


if __name__ == "__main__":
    geoms = []
    for sym in (1400, 1500, 1600, 1700):
        for x0 in range(200, 621, 60):
            for y0 in range(200, 681, 60):
                geoms.append((x0, y0, sym))
    survivors = []
    t0 = time.time()
    for v, e in itertools.product(VERSIONS, ECLS):
        try:
            enc = QArtEncoder(URL, v, e)
        except ValueError:
            continue
        cwmap = codeword_map(enc)
        bases = {mk: enc.base_for_mask(mk) for mk in range(8)}
        enc.base_for_mask = lambda mk, b=bases: b[mk]
        cnt = 0
        for (x0, y0, sym), mk in itertools.product(geoms, range(8)):
            r = stage1(enc, cwmap, mk, x0, y0, sym)
            if r and r["use"] <= MAX_FIXED_USE and r["cover"] >= 0.40:
                survivors.append({"v": v, "e": e, "mask": mk, "x0": x0, "y0": y0, "sym": sym, **r}); cnt += 1
        print(f"v{v}{e}: {cnt} survivors ({time.time()-t0:.0f}s)", flush=True)
    survivors.sort(key=lambda r: (-r["cover"], r["use"]))
    json.dump(survivors, open(L.OUT / "stage1.json", "w"), indent=1)
    print("best stage-1:", survivors[:10])
