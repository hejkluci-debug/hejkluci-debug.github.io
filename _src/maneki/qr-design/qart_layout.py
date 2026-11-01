"""Reference-style artwork: big QR behind the Maneki-neko, cat covering ~60 %.

Geometry follows the user's REFERENCE PDF (A4, 300 dpi): the cat is the print
master (frame 0 of the user's animation) drawn at scale 0.78797 at
(295.3, 191.8) px, the QR sits around x 417..1983, y 371..1936.
The QR content is solved with qr-engine/qart.py so that modules hidden by the
cat match what a decoder sees there.

  python3 qart_layout.py URL [--versions 7 8 9 10] [--ecl L M] [--shift 0]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qr-engine"))
from qart import QArtEncoder, solve_planned, codeword_map
from zx_emul import hybrid_binarize, sample_grid
from qr_engine import decoders

W, H = 2480, 3508
DARK = np.array([38, 0, 3], np.float32)       # BGR navy of the reference QR
LIGHT = np.array([255, 255, 255], np.float32)
CAT_SCALE, CAT_X, CAT_Y = 0.78797, 295.3, 191.8
OUT = ROOT / "builds" / "qart"; OUT.mkdir(parents=True, exist_ok=True)
DOT_R = 0.38
DOT_DARK = (150, 62, 18)      # BGR, the cat's cobalt blue

master = cv2.imread(str(ROOT / "assets/print/maneki_print_master.png"), cv2.IMREAD_UNCHANGED)
A = np.float32([[CAT_SCALE, 0, CAT_X], [0, CAT_SCALE, CAT_Y]])
cat = cv2.warpAffine(master, A, (W, H), flags=cv2.INTER_AREA, borderValue=(0, 0, 0, 0))
cat_rgb = cat[..., :3].astype(np.float32)
cat_a = cat[..., 3].astype(np.float32) / 255.0


def lum(img):
    return img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299


def render_qr(mat, x0, y0, m, style="rounded", finders=None):
    """QR on white canvas; returns float32 BGR."""
    n = mat.shape[0]
    canvas = np.full((H, W, 3), 255, np.uint8)
    col = tuple(int(c) for c in DARK)
    r = int(m * 0.28) if style == "rounded" else 0
    if finders is None:
        finders = [(0, 0), (0, n - 7), (n - 7, 0)]
    finder = lambda y, x: any(fy <= y < fy + 7 and fx <= x < fx + 7 for fy, fx in finders)
    for y in range(n):
        for x in range(n):
            if not mat[y, x] or finder(y, x):
                continue
            X, Y = int(round(x0 + x * m)), int(round(y0 + y * m))
            X2, Y2 = int(round(x0 + (x + 1) * m)) - 1, int(round(y0 + (y + 1) * m)) - 1
            if r:
                cv2.rectangle(canvas, (X + r, Y), (X2 - r, Y2), col, -1)
                cv2.rectangle(canvas, (X, Y + r), (X2, Y2 - r), col, -1)
                for cx, cy in ((X + r, Y + r), (X2 - r, Y + r), (X + r, Y2 - r), (X2 - r, Y2 - r)):
                    cv2.circle(canvas, (cx, cy), r, col, -1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (X, Y), (X2, Y2), col, -1)
    for fy, fx in finders:      # rounded finder patterns
        X, Y = x0 + fx * m, y0 + fy * m
        rr = int(m * 0.9)
        def rrect(a, b, c, d, color, rad):
            a, b, c, d = int(round(a)), int(round(b)), int(round(c)), int(round(d))
            cv2.rectangle(canvas, (a + rad, b), (c - rad, d), color, -1)
            cv2.rectangle(canvas, (a, b + rad), (c, d - rad), color, -1)
            for cx, cy in ((a + rad, b + rad), (c - rad, b + rad), (a + rad, d - rad), (c - rad, d - rad)):
                cv2.circle(canvas, (cx, cy), rad, color, -1, cv2.LINE_AA)
        rrect(X, Y, X + 7 * m - 1, Y + 7 * m - 1, col, rr)
        rrect(X + m, Y + m, X + 6 * m - 1, Y + 6 * m - 1, (255, 255, 255), int(m * 0.5))
        rrect(X + 2 * m, Y + 2 * m, X + 5 * m - 1, Y + 5 * m - 1, col, int(m * 0.6))
    return canvas.astype(np.float32)


def compose(qr_img):
    a = cat_a[..., None]
    return (cat_rgb * a + qr_img * (1 - a)).astype(np.uint8)


_LC = None
def module_stats(x0, y0, m, n):
    """per module: cat coverage at the module centre and the cat luminance there
    (vectorised: resample the module grid at 5x5 samples, use the centre 3x3)."""
    global _LC
    if _LC is None:
        _LC = lum(cat_rgb) * cat_a
    k = 5
    M = np.float32([[m / k, 0, x0 + m / k / 2 - 0.5], [0, m / k, y0 + m / k / 2 - 0.5]])
    a = cv2.warpAffine(cat_a, M, (n * k, n * k), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA)
    la = cv2.warpAffine(_LC, M, (n * k, n * k), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA)
    a = a.reshape(n, k, n, k)[:, 1:4, :, 1:4]
    la = la.reshape(n, k, n, k)[:, 1:4, :, 1:4]
    cov = a.mean(axis=(1, 3))
    L = la.sum(axis=(1, 3)) / np.maximum(1e-6, a.sum(axis=(1, 3)))
    return cov.astype(np.float32), L.astype(np.float32)


def seen_bits(img, x0, y0, m, n, px_per_module=(8, 11, 15)):
    """What a ZXing-style decoder (HybridBinarizer + centre sampling) reads at
    several camera distances (module = 8 / 11 / 15 px).
    Returns (majority bit, reliable = all distances agree)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    pad = int(6 * m)
    ya, yb = max(int(y0) - pad, 0), min(int(y0 + n * m) + pad, g.shape[0])
    xa, xb = max(int(x0) - pad, 0), min(int(x0 + n * m) + pad, g.shape[1])
    g = g[ya:yb, xa:xb]
    reads = []
    for ppm in px_per_module:
        s = ppm / m
        gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gs = cv2.GaussianBlur(gs, (0, 0), 0.6)
        reads.append(sample_grid(hybrid_binarize(gs), (x0 - xa) * s, (y0 - ya) * s, ppm, n))
    reads = np.array(reads)
    out = (reads.sum(0) * 2 > len(px_per_module)).astype(np.int8)
    reliable = (reads.min(0) == reads.max(0)).astype(np.float32)
    return out, reliable


def finder_corners(n, rot):
    """top-left module of each finder in DISPLAY coordinates for np.rot90(canonical, rot)."""
    can = np.zeros((n, n), np.uint8)
    for fy, fx in ((0, 0), (0, n - 7), (n - 7, 0)):
        can[fy, fx] = 1
        can[fy:fy + 7, fx:fx + 7] |= 2
    d = np.rot90(can, rot)
    out = []
    for fy in (0, n - 7):
        for fx in (0, n - 7):
            if d[fy:fy + 7, fx:fx + 7].min() >= 2:
                out.append((fy, fx))
    return out


def finders_free_rot(x0, y0, m, n, rot, margin=1):
    for fy, fx in finder_corners(n, rot):
        X0, Y0 = int(x0 + (fx - margin) * m), int(y0 + (fy - margin) * m)
        X1, Y1 = int(x0 + (fx + 7 + margin) * m), int(y0 + (fy + 7 + margin) * m)
        if cat_a[max(Y0, 0):Y1, max(X0, 0):X1].max() > 0.05:
            return False
    return True


def finders_free(x0, y0, m, n, margin=1):
    for fy, fx in ((0, 0), (0, n - 7), (n - 7, 0)):
        X0, Y0 = int(x0 + (fx - margin) * m), int(y0 + (fy - margin) * m)
        X1, Y1 = int(x0 + (fx + 7 + margin) * m), int(y0 + (fy + 7 + margin) * m)
        if cat_a[max(Y0, 0):Y1, max(X0, 0):X1].max() > 0.05:
            return False
    return True


def draw_marks(img, marks, x0, y0, m):
    out = img.copy()
    for y, x, v in set(marks):
        cx, cy = int(round(x0 + (x + 0.5) * m)), int(round(y0 + (y + 0.5) * m))
        col = DOT_DARK if v else (255, 255, 255)
        cv2.circle(out, (cx, cy), max(2, int(m * DOT_R)), col, -1, cv2.LINE_AA)
    return out


def quick_ok(img, text, x0, y0, sym):
    """3 decoders on a few views; returns (all_ok, zxing_unused_ec)."""
    import zxingcpp
    phone = cv2.resize(img, (1240, 1754), interpolation=cv2.INTER_AREA)
    a, b = int((x0 - 120) / 2), int((y0 - 120) / 2)
    crop = phone[max(b, 0):b + int(sym / 2) + 120, max(a, 0):a + int(sym / 2) + 120]
    views = [phone, crop, cv2.GaussianBlur(crop, (5, 5), 0), cv2.resize(crop, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)]
    ok = all(all(decoders(v, text).values()) for v in views)
    uec = 0.0
    r = [x for x in zxingcpp.read_barcodes(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) if x.text == text]
    if r:
        uec = float(r[0].extra.get("UEC", 0))
    return ok, uec


def prune_dots(base_img, marks, cw, text, x0, y0, m, sym, min_uec=0.45):
    """Remove dots codeword by codeword while every decoder still reads every
    view and zxing still has >= min_uec of its error correction unused."""
    groups = {}
    for (y, x, v) in marks:
        groups.setdefault(int(cw[y, x]) if cw[y, x] >= 0 else -1, []).append((y, x, v))
    keep = set(marks)
    order = sorted([g for g in groups if g >= 0], key=lambda g: -len(groups[g]))
    for g in order:
        trial = keep - set(groups[g])
        ok, uec = quick_ok(draw_marks(base_img, trial, x0, y0, m), text, x0, y0, sym)
        if ok and uec >= min_uec:
            keep = trial
    return keep


def tilt(img, k):
    h, w = img.shape[:2]; d = k * w
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                    np.float32([[d, d * .5], [w - d, 0], [w, h], [0, h - d * .5]]))
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))


def stress(img, text, x0=None, y0=None, sym=None):
    phone = cv2.resize(img, (1240, 1754), interpolation=cv2.INTER_AREA)
    if x0 is None:
        crop = phone[100:1100, 100:1100]
    else:
        a, b = int(x0 / 2) - 80, int(y0 / 2) - 80
        crop = phone[max(b, 0):b + int(sym / 2) + 160, max(a, 0):a + int(sym / 2) + 160]
    tests = {
        "phone": phone,
        "crop": crop,
        "tilt15": tilt(crop, 0.15),
        "blur": cv2.GaussianBlur(crop, (7, 7), 0),
        "dim": (crop.astype(np.float32) * 0.5 + 30).astype(np.uint8),
        "noise": np.clip(crop.astype(np.int16) + np.random.default_rng(2).normal(0, 18, crop.shape), 0, 255).astype(np.uint8),
        "small": cv2.resize(crop, (500, 500), interpolation=cv2.INTER_AREA),
    }
    res = {k: decoders(v, text) for k, v in tests.items()}
    score = sum(sum(v.values()) for v in res.values())
    return score, res


def run(url, version, ecl, mask, x0, y0, sym_px, iters=3, save=False, ec_use=0.6, keep=0.35, prune=False, dots=True, rot=0):
    enc = QArtEncoder(url, version, ecl)
    n = enc.size; m = sym_px / n
    if not finders_free_rot(x0, y0, m, n, rot):
        return None
    Rd = lambda a: np.ascontiguousarray(np.rot90(a, rot))      # canonical -> display
    Uc = lambda a: np.ascontiguousarray(np.rot90(a, -rot))     # display -> canonical
    fc = finder_corners(n, rot)
    isfunc_d = Rd(enc.isfunc)
    cov, L = module_stats(x0, y0, m, n)
    covered = cov > 0.15
    cat_only = compose(np.full((H, W, 3), 255, np.float32))
    seen0, rel0 = seen_bits(cat_only, x0, y0, m, n)
    target = np.full((n, n), -1, np.int8)
    use = covered & (rel0 > 0.5)
    target[use] = seen0[use]
    prio = np.where(use, cov, 0)
    hist = []
    for it in range(iters):
        bits, mat_c, rep = solve_planned(enc, mask, Uc(target), Uc(prio), ec_use)
        mat = Rd(mat_c)
        img = compose(render_qr(mat, x0, y0, m, finders=fc))
        seen, agree = seen_bits(img, x0, y0, m, n)
        wrong = (seen != mat) & ~isfunc_d
        hist.append({"iter": it, **rep, "decoder_view_errors": int(wrong.sum())})
        # next round: aim the covered modules at what the decoder really sees
        target = np.full((n, n), -1, np.int8)
        use = covered & (agree > 0.5)
        target[use] = seen[use]
        prio = np.where(use, cov, 0)
    text = url + "#" + enc.digits(bits)
    # ---- halftone repair dots: wherever the decoder would read a covered
    # module wrong or unreliably, print a small dot of the right colour at its
    # centre (halftone-QR idea, Chu et al. 2013). Function patterns included. ----
    cw, block_of, is_data, data_index, dlens, eclen = codeword_map(enc)
    cw = Rd(cw)
    t = eclen // 2
    marks = set()
    for _ in range(3 if dots else 0):
        seen, rel = seen_bits(img, x0, y0, m, n)
        need = covered & ((seen != mat) | (rel < 0.5))
        new_marks = {(int(y), int(x), int(mat[y, x])) for y, x in zip(*np.where(need))} - marks
        if not new_marks:
            break
        marks |= new_marks
        img = draw_marks(img, marks, x0, y0, m)
    if prune:
        nodot = compose(render_qr(mat, x0, y0, m, finders=fc))
        marks = prune_dots(nodot, marks, cw, text, x0, y0, m, sym_px)
        img = draw_marks(nodot, marks, x0, y0, m)
    ok_uec = quick_ok(img, text, x0, y0, sym_px)
    seen, rel = seen_bits(img, x0, y0, m, n)
    wrong = ((seen != mat) | (rel < 0.5)) & (cw >= 0)
    wrong_cw = {int(cw[y, x]) for y, x in zip(*np.where(wrong))}
    per_block = [sum(1 for c in wrong_cw if block_of[c] == b) for b in range(len(dlens))]
    score, res = stress(img, text, x0, y0, sym_px)
    symbol_cover = float((cat_a[int(y0):int(y0 + sym_px), int(x0):int(x0 + sym_px)] > 0.5).mean())
    out = {"rot": rot, "url": url, "version": version, "ecl": ecl, "mask": mask, "x0": x0, "y0": y0, "sym_px": sym_px,
           "module_mm": round(m / 300 * 25.4, 2), "cover": round(symbol_cover, 3), "score": score,
           "max_score": 21, "hist": hist, "tests": res, "text_len": len(text),
           "marks": len(marks), "dots_dark": sum(1 for _, _, v in marks if v), "dots_light": sum(1 for _, _, v in marks if not v), "wrong_cw_per_block_after_marks": per_block, "t": t, "quick_ok": ok_uec[0], "zxing_unused_ec": ok_uec[1]}
    if save:
        tag = f"v{version}{ecl}m{mask}r{rot}"
        cv2.imwrite(str(OUT / f"qart_{tag}.png"), img)
        np.save(OUT / f"qart_{tag}_matrix.npy", mat)
        json.dump({**out, "text": text}, open(OUT / f"qart_{tag}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--versions", type=int, nargs="+", default=[8])
    ap.add_argument("--ecl", nargs="+", default=["M"])
    ap.add_argument("--masks", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--x0", type=float, default=447); ap.add_argument("--y0", type=float, default=401)
    ap.add_argument("--sym", type=float, default=1506)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--ec_use", type=float, default=0.6)
    ap.add_argument("--prune", action="store_true")
    a = ap.parse_args()
    results = []
    for v in a.versions:
        for e in a.ecl:
            for mk in a.masks:
                t = time.time()
                r = run(a.url, v, e, mk, a.x0, a.y0, a.sym, save=a.save, ec_use=a.ec_use, prune=a.prune)
                if r is None:
                    print(v, e, mk, "finder covered"); continue
                h = r["hist"][-1]
                print(f"v{v} {e} mask{mk}: cover {r['cover']:.0%} module {r['module_mm']} mm | free {h['free_bits']} "
                      f"constraints {h['constraints']} given_up {h['given_up']} wrongCW/blk {h['wrong_codewords_per_block']} (t={h['t_per_block']}) view-err {h['decoder_view_errors']} "
                      f"| quick {r['quick_ok']} UEC {r['zxing_unused_ec']:.2f} | marks {r['marks']} wrongCW-after {r['wrong_cw_per_block_after_marks']} t={r['t']} | decode score {r['score']}/21 ({time.time()-t:.0f}s)", flush=True)
                results.append(r)
    json.dump(results, open(OUT / f"search_{int(time.time())}.json", "w"), indent=1)
