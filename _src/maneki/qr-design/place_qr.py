"""QR design engine v1: place a standard QR next to the protected cat motif.

The cat (its alpha, dilated by a safety margin) is a protected area. The QR
including its 4-module quiet zone must not touch it. The script searches all
positions/sizes on a grid, keeps the largest QR per region (top-left,
top-right, bottom-left, bottom-right, left, right) and renders each variant.
Every composite is then validated by 3 decoders at full size and at a
"phone photo" scale (1080 px wide).

  python3 place_qr.py ../builds/qr_short/qr_H_rounded.png "HTTPS://MNK.CZ/1"
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "qr-engine"))
from qr_engine import decoders

QR_PNG, URL = sys.argv[1], sys.argv[2]
OUT = Path(__file__).parent.parent / "builds" / "layouts"; OUT.mkdir(parents=True, exist_ok=True)
CW, CH = 2400, 3000                # print canvas, 300 dpi -> 20.3 x 25.4 cm
cat = cv2.imread(str(Path(__file__).parent.parent / "animation/out/cat_full.png"), cv2.IMREAD_UNCHANGED)
s = 2550 / cat.shape[0]
cat = cv2.resize(cat, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
ch, cw = cat.shape[:2]
cx, cy = (CW - cw) // 2 + 180, CH - ch - 40          # cat slightly right of centre
protected = np.zeros((CH, CW), np.uint8)
protected[cy:cy + ch, cx:cx + cw] = (cat[..., 3] > 10).astype(np.uint8) * 255
protected = cv2.dilate(protected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
integral = cv2.integral((protected > 0).astype(np.uint8))

def free(x, y, size):
    return integral[y + size, x + size] - integral[y, x + size] - integral[y + size, x] + integral[y, x] == 0

qr = cv2.imread(QR_PNG)                      # includes its quiet zone (white)
margin, best = 60, {}
cat_c = np.array([cx + cw / 2, cy + ch * 0.45])
for size in range(1100, 499, -20):
    for y in range(margin, CH - size - margin + 1, 20):
        for x in range(margin, CW - size - margin + 1, 20):
            if not free(x, y, size):
                continue
            c = np.array([x + size / 2, y + size / 2])
            region = ("top" if c[1] < CH / 3 else "bottom" if c[1] > 2 * CH / 3 else "mid") + "-" + \
                     ("left" if c[0] < CW / 2 else "right")
            score = size - 0.15 * np.linalg.norm(c - cat_c)    # big, but close to the cat
            if region not in best or score > best[region]["score"]:
                best[region] = {"x": x, "y": y, "size": size, "score": float(score)}

results = []
for region, b in sorted(best.items(), key=lambda kv: -kv[1]["score"]):
    canvas = np.full((CH, CW, 3), 255, np.uint8)
    q = cv2.resize(qr, (b["size"], b["size"]), interpolation=cv2.INTER_AREA)
    canvas[b["y"]:b["y"] + b["size"], b["x"]:b["x"] + b["size"]] = q
    al = cat[..., 3:4] / 255.0
    roi = canvas[cy:cy + ch, cx:cx + cw].astype(np.float32)
    canvas[cy:cy + ch, cx:cx + cw] = (cat[..., :3] * al + roi * (1 - al)).astype(np.uint8)
    full = decoders(canvas, URL)
    phone = decoders(cv2.resize(canvas, (1080, int(1080 * CH / CW)), interpolation=cv2.INTER_AREA), URL)
    mm = b["size"] / 300 * 25.4
    modules = int(round(qr.shape[0] / (qr.shape[0] / 33)))  # 25 + 2*4 quiet for version 2
    fn = OUT / f"layout_{region}.png"
    cv2.imwrite(str(fn), canvas)
    results.append({"region": region, **b, "qr_mm_incl_quiet_zone": round(mm, 1),
                    "module_mm": round(mm / modules, 2), "decode_full": full, "decode_phone_scale": phone,
                    "file": fn.name})
    print(region, b, f"{mm:.0f} mm, module {mm/modules:.2f} mm", "full", full, "phone", phone)
json.dump(results, open(OUT / "layouts.json", "w"), indent=1)
