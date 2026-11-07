"""Dynamit T-shirt: big QR in the BACKGROUND behind the explosion illustration
(adapted from Maneki qart_layout.py). Canvas = the user's illustration (300 dpi,
purple background). QR = light panel with modules in the background purple;
illustration composited on top; QArt + halftone repair dots on the art.
"""
import json, sys, os, time
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/home/claude/maneki/qr-engine")
from qart import QArtEncoder, solve_planned, codeword_map
from zx_emul import hybrid_binarize, sample_grid
from qr_engine import decoders

art = cv2.imread(str(ROOT / os.environ.get("ART", "in/ilustrace_rgba.png")), cv2.IMREAD_UNCHANGED)
H, W = art.shape[:2]
BG = np.array([float(v) for v in os.environ.get("BGC", "189,0,136").split(",")], np.float32)        # illustration background purple (BGR)
DARK = BG                                        # QR dark modules = background purple
LIGHT = np.array([255, 228, 196], np.float32)   # light panel (pale smoke blue)
QZ = int(os.environ.get("QZ", 3))                                           # quiet zone in modules
OUT = ROOT / "builds"; OUT.mkdir(exist_ok=True)
DOT_R = 0.38
DOT_DARK = (45, 35, 30)        # BGR, the illustration outline colour
DOT_LIGHT = (255, 228, 196)
STYLE = os.environ.get("STYLE", "panel")       # panel = light panel; dark = dark modules straight on purple (user sketch)
if STYLE == "lightdots":
    DARK = BG.copy(); LIGHT = np.array([255., 255., 255.])
    DOT_DARK = tuple(int(v) for v in os.environ.get("DOTDARK", "189,0,136").split(",")); DOT_LIGHT = (255, 255, 255)
if STYLE == "dark":
    LIGHT = np.array([float(v) for v in os.environ.get("LIGHTC", "189,0,136").split(",")], np.float32)
    DARK = np.array([float(v) for v in os.environ.get("DARKC", "81,0,65").split(",")], np.float32)
    DOT_DARK = tuple(int(v) for v in DARK)
cat_rgb = art[..., :3].astype(np.float32)
cat_a = art[..., 3].astype(np.float32) / 255.0
def lum(img):
    return img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299


def render_qr_lightdots(mat, x0, y0, m, finders=None):
    """User's look: purple background = dark modules, light modules = white round dots,
    black finder patterns with a white ring. Quiet zone = white dots too."""
    n = mat.shape[0]
    canvas = np.empty((H, W, 3), np.uint8); canvas[:] = BG.astype(np.uint8)
    white = tuple(int(c) for c in LIGHTDOT); blk = tuple(int(c) for c in FINDERC)
    if os.environ.get("PANELC"):
        pc = tuple(int(v) for v in os.environ["PANELC"].split(","))
        cv2.rectangle(canvas, (int(x0 - QZ * m), int(y0 - QZ * m)), (int(x0 + (n + QZ) * m), int(y0 + (n + QZ) * m)), pc, -1)
    if finders is None:
        finders = [(0, 0), (0, n - 7), (n - 7, 0)]
    infinder = lambda y, x: any(fy <= y < fy + 7 and fx <= x < fx + 7 for fy, fx in finders)
    sep = lambda y, x: any(fy - 1 <= y < fy + 8 and fx - 1 <= x < fx + 8 for fy, fx in finders)
    rad = DOTSIZE * m / 2
    pal = None
    if os.environ.get("PALETTE"):
        pal = [tuple(int(v) for v in c.split(",")) for c in os.environ["PALETTE"].split(";")]   # BGR colours
        pw = np.array([float(v) for v in os.environ.get("PALW", ",".join(["1"] * len(pal))).split(",")]); pw /= pw.sum()
    def dotcol(y, x):
        if pal is None: return white
        rng = np.random.default_rng((y + 1000) * 7919 + (x + 1000))
        return pal[int(rng.choice(len(pal), p=pw))]
    SQ = os.environ.get("SQ") == "1"      # cat-like rounded squares instead of dots, no dotted quiet zone
    if SQ:
        rq = int(m * 0.28)
        for y in range(n):
            for x in range(n):
                if mat[y, x] or infinder(y, x):
                    continue
                X, Y = int(round(x0 + x * m)), int(round(y0 + y * m))
                X2, Y2 = int(round(x0 + (x + 1) * m)) - 1, int(round(y0 + (y + 1) * m)) - 1
                cv2.rectangle(canvas, (X + rq, Y), (X2 - rq, Y2), white, -1)
                cv2.rectangle(canvas, (X, Y + rq), (X2, Y2 - rq), white, -1)
                for qx, qy in ((X + rq, Y + rq), (X2 - rq, Y + rq), (X + rq, Y2 - rq), (X2 - rq, Y2 - rq)):
                    cv2.circle(canvas, (qx, qy), rq, white, -1, cv2.LINE_AA)
    for y in range(-QZ, n + QZ):
        if SQ: break
        for x in range(-QZ, n + QZ):
            inside = 0 <= y < n and 0 <= x < n
            if inside and (mat[y, x] or infinder(y, x)):
                continue
            cx, cy = x0 + (x + .5) * m, y0 + (y + .5) * m
            if SOLIDSEP and sep(y, x) and not infinder(y, x):
                X, Y = int(round(x0 + x * m)), int(round(y0 + y * m))
                cv2.rectangle(canvas, (X, Y), (int(round(x0 + (x + 1) * m)) - 1, int(round(y0 + (y + 1) * m)) - 1), white, -1)
            else:
                cv2.circle(canvas, (int(round(cx)), int(round(cy))), int(round(rad)), dotcol(y, x), -1, cv2.LINE_AA)
    def rrect(a, b, c, d, color, rr):
        a, b, c, d = int(round(a)), int(round(b)), int(round(c)), int(round(d))
        cv2.rectangle(canvas, (a + rr, b), (c - rr, d), color, -1)
        cv2.rectangle(canvas, (a, b + rr), (c, d - rr), color, -1)
        for qx, qy in ((a + rr, b + rr), (c - rr, b + rr), (a + rr, d - rr), (c - rr, d - rr)):
            cv2.circle(canvas, (qx, qy), rr, color, -1, cv2.LINE_AA)
    if ATOP and CUR_N_ROT[0] == n and os.environ.get("DOTFINDER") != "1":
        for ay, ax in align_blocks(n, CUR_N_ROT[1]):
            X, Y = x0 + ax * m, y0 + ay * m
            pad = 0.5 * m
            rrect(X - pad, Y - pad, X + 5 * m + pad - 1, Y + 5 * m + pad - 1, white, int(m * 0.5))
            rrect(X, Y, X + 5 * m - 1, Y + 5 * m - 1, blk, int(m * 0.5))
            rrect(X + m, Y + m, X + 4 * m - 1, Y + 4 * m - 1, white, int(m * 0.4))
            rrect(X + 2 * m, Y + 2 * m, X + 3 * m - 1, Y + 3 * m - 1, blk, int(m * 0.3))
    if os.environ.get("FSOLID") == "1":
        # standard-looking solid finder / alignment patterns: dark square, white ring, dark centre, white halo
        def rr(a, b, c, d, color, r):
            a, b, c, d = int(round(a)), int(round(b)), int(round(c)), int(round(d))
            cv2.rectangle(canvas, (a + r, b), (c - r, d), color, -1); cv2.rectangle(canvas, (a, b + r), (c, d - r), color, -1)
            for qx, qy in ((a + r, b + r), (c - r, b + r), (a + r, d - r), (c - r, d - r)):
                cv2.circle(canvas, (qx, qy), r, color, -1, cv2.LINE_AA)
        hw = (255, 255, 255)
        for fy, fx in finders:
            X, Y = x0 + fx * m, y0 + fy * m
            rr(X - m, Y - m, X + 8 * m - 1, Y + 8 * m - 1, hw, int(m * 0.8))
            rr(X, Y, X + 7 * m - 1, Y + 7 * m - 1, blk, int(m * 0.7))
            rr(X + m, Y + m, X + 6 * m - 1, Y + 6 * m - 1, hw, int(m * 0.5))
            rr(X + 2 * m, Y + 2 * m, X + 5 * m - 1, Y + 5 * m - 1, blk, int(m * 0.45))
        if ATOP and CUR_N_ROT[0] == n:
            for ay, ax in align_blocks(n, CUR_N_ROT[1]):
                X, Y = x0 + ax * m, y0 + ay * m
                rr(X - .5 * m, Y - .5 * m, X + 5.5 * m - 1, Y + 5.5 * m - 1, hw, int(m * 0.6))
                rr(X, Y, X + 5 * m - 1, Y + 5 * m - 1, blk, int(m * 0.55))
                rr(X + m, Y + m, X + 4 * m - 1, Y + 4 * m - 1, hw, int(m * 0.4))
                rr(X + 2 * m, Y + 2 * m, X + 3 * m - 1, Y + 3 * m - 1, blk, int(m * 0.3))
        return canvas.astype(np.float32)
    if os.environ.get("DOTFINDER") == "1":
        # finder / alignment patterns drawn as rings of dots (light modules only); dark parts = background
        frad = float(os.environ.get("FDOT", DOTSIZE)) * m / 2
        def dot(yy, xx):
            cv2.circle(canvas, (int(round(x0 + (xx + .5) * m)), int(round(y0 + (yy + .5) * m))), int(round(frad)), white, -1, cv2.LINE_AA)
        for fy, fx in finders:
            for k in range(7):                       # light ring 5x5 inside the finder
                for (yy, xx) in ((fy + 1, fx + 1 + k), (fy + 5, fx + 1 + k), (fy + 1 + k, fx + 1), (fy + 1 + k, fx + 5)):
                    if fy + 1 <= yy <= fy + 5 and fx + 1 <= xx <= fx + 5: dot(yy, xx)
            for k in range(-1, 8):                   # separator / halo ring around the finder (also outside the symbol)
                for (yy, xx) in ((fy - 1, fx + k), (fy + 7, fx + k), (fy + k, fx - 1), (fy + k, fx + 7)):
                    dot(yy, xx)
        if ATOP and CUR_N_ROT[0] == n:
            for ay, ax in align_blocks(n, CUR_N_ROT[1]):
                for (yy, xx) in ((ay + 1, ax + 1), (ay + 1, ax + 2), (ay + 1, ax + 3), (ay + 3, ax + 1), (ay + 3, ax + 2), (ay + 3, ax + 3), (ay + 2, ax + 1), (ay + 2, ax + 3)):
                    dot(yy, xx)
        return canvas.astype(np.float32)
    for fy, fx in finders:
        X, Y = x0 + fx * m, y0 + fy * m
        def rrect(a, b, c, d, color, rr):
            a, b, c, d = int(round(a)), int(round(b)), int(round(c)), int(round(d))
            cv2.rectangle(canvas, (a + rr, b), (c - rr, d), color, -1)
            cv2.rectangle(canvas, (a, b + rr), (c, d - rr), color, -1)
            for qx, qy in ((a + rr, b + rr), (c - rr, b + rr), (a + rr, d - rr), (c - rr, d - rr)):
                cv2.circle(canvas, (qx, qy), rr, color, -1, cv2.LINE_AA)
        if os.environ.get("SQ") == "1":
            hz = float(os.environ.get("HALO", 1.0))
            rrect(X - hz * m, Y - hz * m, X + (7 + hz) * m - 1, Y + (7 + hz) * m - 1, white, int(m * 0.9))
        rrect(X, Y, X + 7 * m - 1, Y + 7 * m - 1, blk, int(m * 0.6))
        rrect(X + m, Y + m, X + 6 * m - 1, Y + 6 * m - 1, white, int(m * 0.4))
        rrect(X + 2 * m, Y + 2 * m, X + 5 * m - 1, Y + 5 * m - 1, blk, int(m * 0.5))
    return canvas.astype(np.float32)

LIGHTDOT = np.array([float(v) for v in os.environ.get("LIGHTDOT", "255,255,255").split(",")])
FINDERC = np.array([float(v) for v in os.environ.get("FINDERC", "30,10,25").split(",")])
DOTSIZE = float(os.environ.get("DOTSIZE", 0.86))
SOLIDSEP = os.environ.get("SOLIDSEP", "0") == "1"

def render_qr(mat, x0, y0, m, style="rounded", finders=None):
    """QR on white canvas; returns float32 BGR."""
    if STYLE == "lightdots":
        return render_qr_lightdots(mat, x0, y0, m, finders)
    n = mat.shape[0]
    canvas = np.empty((H, W, 3), np.uint8); canvas[:] = BG.astype(np.uint8)
    col = tuple(int(c) for c in DARK); lc = tuple(int(c) for c in LIGHT)
    S0 = x0 - QZ * m; T0 = y0 - QZ * m; S1 = x0 + (n + QZ) * m; T1 = y0 + (n + QZ) * m
    rp = int(m * 1.2)
    if STYLE == "dark": rp = 0
    cv2.rectangle(canvas, (int(S0 + rp), int(T0)), (int(S1 - rp), int(T1)), lc, -1)
    cv2.rectangle(canvas, (int(S0), int(T0 + rp)), (int(S1), int(T1 - rp)), lc, -1)
    for cx, cy in ((S0 + rp, T0 + rp), (S1 - rp, T0 + rp), (S0 + rp, T1 - rp), (S1 - rp, T1 - rp)):
        cv2.circle(canvas, (int(cx), int(cy)), rp, lc, -1, cv2.LINE_AA)
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
        rrect(X + m, Y + m, X + 6 * m - 1, Y + 6 * m - 1, lc, int(m * 0.5))
        rrect(X + 2 * m, Y + 2 * m, X + 5 * m - 1, Y + 5 * m - 1, col, int(m * 0.6))
    return apply_glow(canvas, m).astype(np.float32)


FMASK = None          # finder patterns drawn ON TOP of the art (env FTOP=1)
NEAR = int(os.environ.get("NEAR", 0))           # also steer/repair modules this many modules around the art
GLOW = None
def glow_map(m):
    """Lighter purple only near the art (a glow), plain background purple further away."""
    global GLOW
    gi, go = (float(v) for v in os.environ.get("GLOW", "2,4").split(","))
    art_mask = (cat_a > 0.15).astype(np.uint8)
    dist = cv2.distanceTransform(1 - art_mask, cv2.DIST_L2, 5)
    w = np.clip((go * m - dist) / max((go - gi) * m, 1), 0, 1)
    if FMASK is not None and os.environ.get("FHALO") == "1":      # lighter halo behind the 3 finder patterns
        fd = cv2.distanceTransform((~FMASK).astype(np.uint8), cv2.DIST_L2, 5)
        w = np.maximum(w, np.clip((2 * m - fd) / m, 0, 1))
    w = w[..., None]
    GLOW = (BG * (1 - w) + LIGHT * w).astype(np.uint8)
    return GLOW

ALIGN_POS = {2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42],
             9: [6, 26, 46], 10: [6, 28, 50]}
def align_blocks(n, rot):
    """Top-left (y, x) of 5x5 alignment patterns in display coordinates."""
    v = (n - 17) // 4; ps = ALIGN_POS.get(v, []); out = []
    for cy in ps:
        for cx in ps:
            if (cy == 6 and cx == 6) or (cy == 6 and cx == ps[-1]) or (cy == ps[-1] and cx == 6):
                continue
            y, x = cy - 2, cx - 2
            for _ in range(rot % 4):
                y, x = n - 5 - x, y
            out.append((y, x))
    if os.environ.get("ATOPONE") == "1" and out:
        out = [max(out, key=lambda t: t[0] + t[1])] if rot % 4 == 0 else out[-1:]
    return out
ATOP = os.environ.get("ATOP") == "1"
CUR_N_ROT = [None, 0]
def set_fmask(x0, y0, m, n, rot):
    global FMASK
    CUR_N_ROT[0], CUR_N_ROT[1] = n, rot
    FMASK = np.zeros((H, W), bool)
    if ATOP:
        for ay, ax in align_blocks(n, rot):
            X0, Y0 = int(x0 + (ax - .5) * m), int(y0 + (ay - .5) * m)
            X1, Y1 = int(x0 + (ax + 5.5) * m), int(y0 + (ay + 5.5) * m)
            FMASK[max(Y0, 0):Y1, max(X0, 0):X1] = True
    for fy, fx in finder_corners(n, rot):
        X0, Y0 = int(x0 + (fx - 1) * m), int(y0 + (fy - 1) * m)
        X1, Y1 = int(x0 + (fx + 8) * m), int(y0 + (fy + 8) * m)
        FMASK[max(Y0, 0):Y1, max(X0, 0):X1] = True

def apply_glow(canvas, m):
    if not os.environ.get("GLOW"):
        return canvas
    g = GLOW if GLOW is not None else glow_map(m)
    light = np.all(canvas == LIGHT.astype(np.uint8), axis=2)
    canvas[light] = g[light]
    return canvas

def compose(qr_img):
    a = cat_a[..., None]
    out = (cat_rgb * a + qr_img * (1 - a)).astype(np.uint8)
    if FMASK is not None:
        out[FMASK] = qr_img[FMASK].astype(np.uint8)
    return out


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
        col = DOT_DARK if v else DOT_LIGHT
        cv2.circle(out, (cx, cy), max(2, int(m * DOT_R)), col, -1, cv2.LINE_AA)
    return out


def quick_ok(img, text, x0, y0, sym):
    """3 decoders on a few views; returns (all_ok, zxing_unused_ec)."""
    import zxingcpp
    phone = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
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
    return cv2.warpPerspective(img, M, (w, h), borderValue=tuple(int(c) for c in BG))


def stress(img, text, x0=None, y0=None, sym=None):
    phone = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if x0 is None:
        crop = phone
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
    if os.environ.get("FTOP") == "1":
        set_fmask(x0, y0, m, n, rot)
    elif not finders_free_rot(x0, y0, m, n, rot):
        return None
    Rd = lambda a: np.ascontiguousarray(np.rot90(a, rot))      # canonical -> display
    Uc = lambda a: np.ascontiguousarray(np.rot90(a, -rot))     # display -> canonical
    fc = finder_corners(n, rot)
    isfunc_d = Rd(enc.isfunc)
    cov, L = module_stats(x0, y0, m, n)
    covered = cov > 0.15
    if NEAR:
        covered = cv2.dilate(covered.astype(np.uint8), np.ones((2 * NEAR + 1, 2 * NEAR + 1), np.uint8)).astype(bool)
    fz = np.zeros((n, n), bool)
    for fy, fx in finder_corners(n, rot):
        fz[max(fy - 1, 0):fy + 8, max(fx - 1, 0):fx + 8] = True
    covered &= ~fz
    cat_only = compose(render_qr(np.zeros((n, n), np.uint8), x0, y0, m, finders=[]))
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

    out["_img"] = img; out["_mat"] = mat; out["text"] = text
    return out


