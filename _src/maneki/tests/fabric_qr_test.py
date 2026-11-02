"""Decode test of a print after simulated DTF-on-cotton printing and phone capture."""
import sys, json
import numpy as np, cv2
sys.path.insert(0, '../qr-engine')
from qr_engine import decoders

def fabric(img, rng):
    h, w = img.shape[:2]
    fab = np.clip(232 + rng.normal(0, 6, (h, w)), 0, 255).astype(np.float32)[..., None]
    a = img.astype(np.float32) / 255
    out = fab * (0.12 + 0.88 * a)
    return out

def run(png, text, box, label):
    global msize_px
    img = cv2.imread(png); rng = np.random.default_rng(int(__import__("os").environ.get("SEED","3")))
    x0, y0, x1, y1 = box
    f = fabric(img, rng)
    res = {}
    for ppm, msize in ((6, None), (9, None), (12, None)):
        crop = f[y0:y1, x0:x1]
        s = ppm / msize_px
        im = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for light in (0.8, 0.55):
            v = np.clip(im * light + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
            v = cv2.GaussianBlur(v, (3, 3), 0.8)
            d = decoders(v, text)
            res[f"{ppm}px_light{light}"] = "".join("1" if x else "0" for x in d.values())
    print(label, res)

if __name__ == "__main__":
    c = json.load(open('../assets/print/maneki_C_qr.json'))
    n = 4 * c['v'] + 17; msize_px = c['sym'] / n
    pad = int(6 * msize_px)
    run('../assets/print/maneki_C_print_A4_300dpi.png', c['text'], (c['x0'] - pad, c['y0'] - pad, c['x0'] + c['sym'] + pad, c['y0'] + c['sym'] + pad), 'C')
    a = json.load(open('../builds/layouts/layout_behind_gh.json'))['best']
    msize_px = a['size'] / 37
    run('../builds/layouts/layout_behind_gh.png', 'HTTPS://HEJKLUCI-DEBUG.GITHUB.IO/M/', (a['x'], a['y'], a['x'] + a['size'], a['y'] + a['size']), 'A')
