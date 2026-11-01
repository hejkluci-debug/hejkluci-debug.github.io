"""QR engine v1: generate, style, export and validate standard QR codes.

  python3 qr_engine.py "HTTPS://MNK.CZ/1" --out ../builds/qr

For each EC level it reports version, mode and size, writes SVG + PNG in
the requested style and validates the PNG with three independent decoders
(zxing-cpp, ZBar, OpenCV WeChat) under several distortions.
Nothing here breaks the QR standard: only module shape and colour change,
finder patterns stay solid and the quiet zone stays 4 modules.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import segno
import zxingcpp
from pyzbar import pyzbar
from PIL import Image

WECHAT = cv2.wechat_qrcode.WeChatQRCode()


def decoders(img, expected):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = {"zxing": any(b.text == expected for b in zxingcpp.read_barcodes(g)),
         "zbar": any(d.data.decode(errors="replace") == expected for d in pyzbar.decode(Image.fromarray(g))),
         "wechat": expected in WECHAT.detectAndDecode(g)[0]}
    return r


def matrix(qr):
    return np.array([[1 if v else 0 for v in row] for row in qr.matrix], np.uint8)


def is_finder(y, x, n):
    return (y < 7 and x < 7) or (y < 7 and x >= n - 7) or (y >= n - 7 and x < 7)


def svg(m, style="rounded", fg="#1d3c8f", bg="#ffffff", qz=4, module=10, dot=0.86):
    n = m.shape[0]; size = (n + 2 * qz) * module
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">',
           f'<rect width="{size}" height="{size}" fill="{bg}"/>', f'<g fill="{fg}">']
    # finder patterns drawn as solid rounded squares (ratio 1:1:3:1:1 kept)
    for fy, fx in ((0, 0), (0, n - 7), (n - 7, 0)):
        X, Y = (fx + qz) * module, (fy + qz) * module
        r = module * 1.2 if style != "square" else 0
        out.append(f'<path fill-rule="evenodd" d="M{X+r} {Y}h{7*module-2*r}a{r} {r} 0 0 1 {r} {r}v{7*module-2*r}a{r} {r} 0 0 1 {-r} {r}h{-(7*module-2*r)}a{r} {r} 0 0 1 {-r} {-r}v{-(7*module-2*r)}a{r} {r} 0 0 1 {r} {-r}z'
                   f'M{X+module} {Y+module}h{5*module}v{5*module}h{-5*module}z"/>')
        out.append(f'<rect x="{X+2*module}" y="{Y+2*module}" width="{3*module}" height="{3*module}" rx="{0.6*module if style!="square" else 0}"/>')
    for y in range(n):
        for x in range(n):
            if not m[y, x] or is_finder(y, x, n):
                continue
            X, Y = (x + qz) * module, (y + qz) * module
            if style == "square":
                out.append(f'<rect x="{X}" y="{Y}" width="{module}" height="{module}"/>')
            elif style == "dots":
                out.append(f'<circle cx="{X+module/2}" cy="{Y+module/2}" r="{module*dot/2:.2f}"/>')
            else:  # rounded squares
                d = module * 0.92; o = (module - d) / 2
                out.append(f'<rect x="{X+o:.2f}" y="{Y+o:.2f}" width="{d:.2f}" height="{d:.2f}" rx="{module*0.3:.2f}"/>')
    out.append("</g></svg>")
    return "\n".join(out)


def svg_to_png(svg_text, path, px):
    tmp = Path(path).with_suffix(".tmp.svg"); tmp.write_text(svg_text)
    import subprocess
    subprocess.run(["rsvg-convert", "-w", str(px), "-h", str(px), str(tmp), "-o", str(path)], check=True)
    tmp.unlink()


def stress(img, expected):
    h, w = img.shape[:2]; res = {}
    res["clean"] = decoders(img, expected)
    for k in (0.15, 0.3):
        d = k * w
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                        np.float32([[d, d * .5], [w - d, 0], [w, h], [0, h - d * .5]]))
        res[f"tilt_{k}"] = decoders(cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255)), expected)
    res["blur"] = decoders(cv2.GaussianBlur(img, (9, 9), 0), expected)
    res["dark_lowcontrast"] = decoders((img.astype(np.float32) * 0.45 + 40).astype(np.uint8), expected)
    res["small_80px"] = decoders(cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA), expected)
    noise = np.clip(img.astype(np.int16) + np.random.default_rng(1).normal(0, 25, img.shape), 0, 255).astype(np.uint8)
    res["noise"] = decoders(noise, expected)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--out", default="out")
    ap.add_argument("--style", default="rounded", choices=["square", "rounded", "dots"])
    ap.add_argument("--fg", default="#1d3c8f")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = []
    for ec in "LMQH":
        qr = segno.make_qr(a.url, error=ec, boost_error=False)
        mode = qr.mode
        m = matrix(qr)
        s = svg(m, a.style, a.fg)
        name = f"qr_{ec}_{a.style}"
        (out / f"{name}.svg").write_text(s)
        svg_to_png(s, out / f"{name}.png", 600)
        img = cv2.imread(str(out / f"{name}.png"))
        res = stress(img, a.url)
        passed = sum(all(v.values()) for v in res.values())
        report.append({"ec": ec, "version": qr.version, "modules": m.shape[0], "mode": mode,
                       "tests_all_decoders_ok": f"{passed}/{len(res)}", "detail": res})
        print(f"EC {ec}: version {qr.version} ({m.shape[0]}x{m.shape[0]}), mode {mode}, "
              f"all 3 decoders ok in {passed}/{len(res)} tests")
    json.dump(report, open(out / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
