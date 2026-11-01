import sys, json, itertools
import numpy as np
sys.path.insert(0, '../qr-engine')
import qart_layout as L
from qart import QArtEncoder, codeword_map
URL = "HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/"
out = []
for v, e in ((9, 'M'), (11, 'M'), (13, 'M'), (11, 'Q')):
    enc = QArtEncoder(URL, v, e); n = enc.size
    cw, bo, isd, di, dl, el = codeword_map(enc)
    free_cw = set(np.array(enc.free) // 8)
    fixed = np.zeros((n, n), bool)
    for c in range(cw.max() + 1):
        if isd[c] and di[c] not in free_cw:
            fixed |= cw == c
    t = el // 2
    for rot in (1, 2):
        fd = np.rot90(fixed, rot); cwd = np.rot90(cw, rot)
        for sym in range(1300, 2101, 100):
            m = sym / n
            for y0 in range(60, 1500, 60):
                for x0 in range(20, 2480 - sym, 60):
                    if not L.finders_free_rot(x0, y0, m, n, rot):
                        continue
                    cov, _ = L.module_stats(x0, y0, m, n)
                    c = cov > 0.15
                    bad_cw = len({int(cwd[a, b]) for a, b in zip(*np.where(c & fd))})
                    cover = float(c.mean())
                    out.append({"v": v, "e": e, "rot": rot, "x0": x0, "y0": y0, "sym": sym, "cover": round(cover, 3), "fixed_cw_covered": bad_cw, "t": t})
    print(v, e, len(out), flush=True)
json.dump(out, open(L.OUT / "scan_rot.json", "w"))
good = [o for o in out if o["fixed_cw_covered"] <= 0.3 * o["t"]]
good.sort(key=lambda o: -o["cover"])
for g in good[:15]: print(g)
