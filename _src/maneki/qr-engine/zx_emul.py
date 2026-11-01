"""Emulation of ZXing's HybridBinarizer + grid sampling, used to see which
modules a real decoder reads wrong (zxing-cpp does not expose its bit matrix)."""
import numpy as np

BLOCK = 8


def hybrid_binarize(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.int32)
    h, w = g.shape
    bh, bw = (h + BLOCK - 1) // BLOCK, (w + BLOCK - 1) // BLOCK
    pad = np.pad(g, ((0, bh * BLOCK - h), (0, bw * BLOCK - w)), mode="edge")
    blocks = pad.reshape(bh, BLOCK, bw, BLOCK)
    s = blocks.sum(axis=(1, 3)); mn = blocks.min(axis=(1, 3)); mx = blocks.max(axis=(1, 3))
    avg = s >> 6
    bp = np.zeros((bh, bw), np.int32)
    for y in range(bh):
        for x in range(bw):
            a = avg[y, x]
            if mx[y, x] - mn[y, x] <= 24:
                a = mn[y, x] // 2
                if y > 0 and x > 0:
                    nb = (bp[y - 1, x] + 2 * bp[y, x - 1] + bp[y - 1, x - 1]) // 4
                    if mn[y, x] < nb:
                        a = nb
            bp[y, x] = a
    # threshold per block = mean of 5x5 neighbourhood of black points
    padbp = np.pad(bp, 2, mode="edge")
    thr = np.zeros_like(bp)
    for dy in range(5):
        for dx in range(5):
            thr += padbp[dy:dy + bh, dx:dx + bw]
    thr //= 25
    T = np.repeat(np.repeat(thr, BLOCK, 0), BLOCK, 1)[:h, :w]
    return (g <= T).astype(np.uint8)          # 1 = black


def sample_grid(bw: np.ndarray, x0, y0, m, n):
    ys = (y0 + (np.arange(n) + 0.5) * m).astype(int)
    xs = (x0 + (np.arange(n) + 0.5) * m).astype(int)
    return bw[np.ix_(ys, xs)]
