"""QR design engine v1 - variant "QR in the background behind the cat".

Closest standard-compatible version of the reference artwork: a large QR sits
behind the cat and the cat may overlap part of it, but only
  * never the three finder patterns (+ separators) or the format information,
  * at most MAX_COVER of the symbol area (EC level H),
  * and only if all three decoders read the composite at phone-photo scale,
    with 15 % perspective tilt and with blur.
The search tries QR sizes and positions and keeps the best-scoring survivors.
"""
import json, os, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "qr-engine"))
from qr_engine import decoders

QR_PNG, URL, VERSION_MODULES = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 25
MAX_COVER = 0.12
OUT = Path(__file__).parent.parent / "builds" / "layouts"; OUT.mkdir(parents=True, exist_ok=True)
CW, CH = 2400, 3000
cat = cv2.imread(os.environ.get("CAT", str(Path(__file__).parent.parent / "animation/out/cat_full.png")), cv2.IMREAD_UNCHANGED)
TAG = os.environ.get("TAG", "")
qr = cv2.imread(QR_PNG)
TOT = VERSION_MODULES + 8

def render(size, x, y, cat_h, cx, cy):
    c = cv2.resize(cat, None, fx=cat_h / cat.shape[0], fy=cat_h / cat.shape[0], interpolation=cv2.INTER_AREA)
    canvas = np.full((CH, CW, 3), 255, np.uint8)
    canvas[y:y + size, x:x + size] = cv2.resize(qr, (size, size), interpolation=cv2.INTER_AREA)
    h, w = c.shape[:2]
    al = c[..., 3:4] / 255.0
    roi = canvas[cy:cy + h, cx:cx + w].astype(np.float32)
    canvas[cy:cy + h, cx:cx + w] = (c[..., :3] * al + roi * (1 - al)).astype(np.uint8)
    mask = np.zeros((CH, CW), bool); mask[cy:cy + h, cx:cx + w] = c[..., 3] > 10
    return canvas, mask

def tilt(img, k=0.15):
    h, w = img.shape[:2]; d = k * w
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                    np.float32([[d, d * .5], [w - d, 0], [w, h], [0, h - d * .5]]))
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

results = []
cat_h = 2450
for size in (1500, 1400, 1300, 1200):
    mod = size / TOT
    for y in range(40, 700, 60):
        for x in range(40, 900, 60):
            cx, cy = CW - int(cat.shape[1] * cat_h / cat.shape[0]) - 20, CH - cat_h - 30
            canvas, mask = render(size, x, y, cat_h, cx, cy)
            sym = mask[y + int(4 * mod):y + size - int(4 * mod), x + int(4 * mod):x + size - int(4 * mod)]
            cover = sym.mean()
            if cover < 0.03 or cover > MAX_COVER:
                continue
            # finder patterns incl. separator (8x8 modules) + format info strips must stay visible
            q0 = x + 4 * mod
            protected_boxes = [(q0, y + 4 * mod), (q0 + (VERSION_MODULES - 8) * mod, y + 4 * mod),
                               (q0, y + 4 * mod + (VERSION_MODULES - 8) * mod)]
            if any(mask[int(by):int(by + 9 * mod), int(bx):int(bx + 9 * mod)].any() for bx, by in protected_boxes):
                continue
            phone = cv2.resize(canvas, (1080, 1350), interpolation=cv2.INTER_AREA)
            ok = all(all(decoders(img, URL).values()) for img in
                     (phone, tilt(phone), cv2.GaussianBlur(phone, (5, 5), 0)))
            if ok:
                results.append({"size": size, "x": x, "y": y, "cover": round(float(cover), 3),
                                "score": size / 1500 + cover})
    if results:
        break
results.sort(key=lambda r: -r["score"])
print(len(results), "survivors; best:", results[:3])
best = results[0]
cx, cy = CW - int(cat.shape[1] * cat_h / cat.shape[0]) - 20, CH - cat_h - 30
canvas, _ = render(best["size"], best["x"], best["y"], cat_h, cx, cy)
cv2.imwrite(str(OUT / f"layout_behind{TAG}.png"), canvas)
json.dump({"best": best, "all": results[:20], "cat": {"h": cat_h, "x": cx, "y": cy}},
          open(OUT / f"layout_behind{TAG}.json", "w"), indent=1)
