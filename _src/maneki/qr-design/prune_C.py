"""Remove repair dots codeword by codeword while all 3 decoders still read all
11 test views (21/21) and zxing keeps >= 30 % of its error correction unused."""
import sys, json, time
import numpy as np, cv2, zxingcpp
import qart_layout as L
from search_C import fitness
from qart import QArtEncoder, codeword_map
import os
URL = os.environ.get("QR_URL", "HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/")
v, e, mk, x0, y0, sym = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
L.DOT_R = 0.38
n = 4 * v + 17; m = sym / n
ROT = int(os.environ.get("ROT", 0))
r = L.run(URL, v, e, mk, x0, y0, sym, dots=True, rot=ROT)
mat, text = r["_mat"], r["text"]
nodot = L.compose(L.render_qr(mat, x0, y0, m, finders=L.finder_corners(n, ROT)))
# recover the mark list: modules where the dotted image differs from nodot at module centres
diff = np.abs(r["_img"].astype(int) - nodot.astype(int)).sum(2) > 30
marks = set()
for y in range(n):
    for x in range(n):
        cx, cy = int(x0 + (x + .5) * m), int(y0 + (y + .5) * m)
        if diff[cy, cx]:
            marks.add((y, x, int(mat[y, x])))
FULL = fitness(r["_img"], text, x0, y0, sym, n)[0]
print("dots", len(marks), "fitness", FULL, fitness(r["_img"], text, x0, y0, sym, n)[1])
enc = QArtEncoder(URL, v, e); cw = np.ascontiguousarray(np.rot90(codeword_map(enc)[0], ROT))
groups = {}
for mrk in marks:
    groups.setdefault(int(cw[mrk[0], mrk[1]]), []).append(mrk)

def uec(img):
    s = 12 / m
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rr = [b for b in zxingcpp.read_barcodes(g) if b.text == text]
    return float(rr[0].extra.get("UEC", 0)) if rr else 0.0

keep = set(marks); t0 = time.time()
order = sorted([g for g in groups if g >= 0], key=lambda g: -len(groups[g]))
for i, g in enumerate(order):
    trial = keep - set(groups[g])
    img = L.draw_marks(nodot, trial, x0, y0, m)
    f, _ = fitness(img, text, x0, y0, sym, n)
    if f >= FULL and uec(img) >= 0.30:
        keep = trial
    if i % 10 == 0:
        print(f"{time.time()-t0:4.0f}s group {i}/{len(order)} dots {len(keep)}", flush=True)
img = L.draw_marks(nodot, keep, x0, y0, m)
f, res = fitness(img, text, x0, y0, sym, n)
tag = os.environ.get("TAG", f"C_v{v}{e}m{mk}")
cv2.imwrite(str(L.OUT / f"{tag}_pruned.png"), img)
np.save(L.OUT / f"{tag}_matrix.npy", mat)
json.dump({"rot": ROT, "v": v, "e": e, "mask": mk, "x0": x0, "y0": y0, "sym": sym, "text": text, "dots": len(keep),
           "dots_before": len(marks), "fitness": f, "res": res, "uec12": uec(img), "cover": r["cover"],
           "marks": sorted(keep)}, open(L.OUT / f"{tag}_pruned.json", "w"))
print("final dots", len(keep), "fitness", f, res, "uec", uec(img))
