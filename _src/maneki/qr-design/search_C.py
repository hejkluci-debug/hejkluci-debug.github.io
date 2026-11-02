"""Search for the most robust reference-style (~60-66 % covered) variant C."""
import sys, json, itertools, time
import numpy as np, cv2
import qart_layout as L
from qr_engine import decoders
URL = "HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/"

def fitness(img, text, x0, y0, sym, n):
    g = img
    a, b = int(x0 - 2 * sym / n * 6), int(y0 - 2 * sym / n * 6)
    crop = g[max(b, 0):int(y0 + sym) + 200, max(a, 0):int(x0 + sym) + 200]
    tot, res = 0, {}
    for ppm in (6, 9, 12, 16):
        s = ppm / (sym / n)
        im = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        views = {"plain": im}
        if ppm in (9, 12):
            rng = np.random.default_rng(ppm)
            fab = np.clip(232 + rng.normal(0, 6, im.shape[:2]), 0, 255).astype(np.float32)[..., None]
            fb = fab * (0.12 + 0.88 * im.astype(np.float32) / 255)
            for light in (0.8, 0.55):
                v = np.clip(fb * light + rng.normal(0, 4, fb.shape), 0, 255).astype(np.uint8)
                views[f"fabric{light}"] = cv2.GaussianBlur(v, (3, 3), 0.8)
        if ppm == 12:
            views["tilt"] = L.tilt(im, 0.15)
            views["blur"] = cv2.GaussianBlur(im, (5, 5), 0)
            views["dim"] = (im.astype(np.float32) * 0.55 + 25).astype(np.uint8)
        for k, v in views.items():
            d = decoders(v, text)
            res[f"{ppm}{k}"] = "".join("1" if x else "0" for x in d.values())
            tot += sum(d.values())
    # full-resolution textile simulation (noise before downscaling), 3 seeds
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        h, w = crop.shape[:2]
        fab = np.clip(232 + rng.normal(0, 6, (h, w)), 0, 255).astype(np.float32)[..., None]
        fb = fab * (0.12 + 0.88 * crop.astype(np.float32) / 255)
        for ppm in (6, 9):
            s = ppm / (sym / n)
            im = cv2.resize(fb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            v = np.clip(im * 0.8 + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
            v = cv2.GaussianBlur(v, (3, 3), 0.8)
            d = decoders(v, text)
            res[f"tex{seed}_{ppm}"] = "".join("1" if x else "0" for x in d.values())
            tot += sum(d.values())
    return tot, res

if __name__ == "__main__":
    geoms = [(340, 520, 1540), (300, 480, 1600), (380, 560, 1480)]
    out = []
    t0 = time.time()
    for (x0, y0, sym), v, e in itertools.product(geoms, (9, 10, 11, 12), ("M", "Q")):
        for mk in range(8):
            for dr in (0.38,):
                L.DOT_R = dr
                r = L.run(URL, v, e, mk, x0, y0, sym, dots=True, rot=0)
                if r is None:
                    continue
                f, res = fitness(r["_img"], r["text"], x0, y0, sym, 4 * v + 17)
                row = {"x0": x0, "y0": y0, "sym": sym, "v": v, "e": e, "mask": mk, "dot_r": dr, "cover": r["cover"],
                       "marks": r["marks"], "fit": f, "max": 21, "res": res}
                out.append(row)
                print(f"{time.time()-t0:5.0f}s", row, flush=True)
                json.dump(out, open(L.OUT / "search_C.json", "w"), indent=1)
