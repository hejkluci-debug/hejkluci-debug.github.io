"""Dynamit 2.0 – AR overlay video from the black-background animation.

1. key out the pure-black background (dark outlines of the drawing stay opaque),
2. warp every frame into PRINT coordinates (video -> PNG registration from SIFT, PNG -> print = 80 % about the centre),
3. encode stacked-alpha (colour on top, alpha below) H.264 MP4 + VP9 WebM at native resolution.

Outputs (in ./out): dyn_anim.mp4, dyn_anim.webm, overlay.json (k, ox, oy, size), preview PNGs.
"""
import json, subprocess
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
SRC = HERE / "dynamit_cerne_pozadi.mp4"
M = np.load(HERE / "M_video_to_png.npy")                 # video -> PNG (similarity)
PW, PH = 3802, 4767                                       # print canvas
ART_C = np.array([1770.0, 2480.5]); PRINT_C = np.array([PW / 2, PH / 2]); ART_S = 0.8

# video -> print
A = ART_S * M[:, :2]
a_off = ART_S * (M[:, 2] - ART_C) + PRINT_C
k = float(np.hypot(A[0, 0], A[1, 0]))                    # print px per video px (keeps native resolution)

cap = cv2.VideoCapture(str(SRC))
fps = cap.get(cv2.CAP_PROP_FPS); vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
mx_all = np.max([f.max(2) for f in frames[::4]], axis=0)
ys, xs = np.where(mx_all > 12)
corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], float)
pc = corners @ A.T + a_off
O = pc.min(0) - 4 * k
ext = (pc.max(0) + 4 * k - O) / k
VW, VH = int(np.ceil(ext[0] / 2) * 2), int(np.ceil(ext[1] / 2) * 2)
B = np.hstack([A / k, ((a_off - O) / k)[:, None]])      # video -> V (V = (print - O) / k)
print("k", k, "O", O, "V size", VW, VH, "frames", len(frames), "fps", fps)

def key(f):
    """Opaque where the drawing is solid; elsewhere alpha from brightness (glow, fading puffs = premultiplied on black)."""
    m = f.max(2)
    core = (m > 150).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    inv = (1 - core).copy(); h, w = inv.shape
    ff = inv.copy(); msk = np.zeros((h + 2, w + 2), np.uint8); cv2.floodFill(ff, msk, (0, 0), 2)
    filled = core | ((ff == 1).astype(np.uint8))                       # holes inside the drawing
    solid = (cv2.dilate(filled, np.ones((7, 7), np.uint8)) > 0) & (m > 8) | (filled > 0)
    a = np.clip(m.astype(np.float32) / 200.0, 0, 1)
    a[solid] = 1.0
    # outer anti-aliased outline: smooth ramp on dark pixels right next to the solid area
    ring = (cv2.dilate(solid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~solid
    a[ring] = np.maximum(a[ring], np.clip((m[ring].astype(np.float32) - 4) / 26.0, 0, 1))
    a[m <= 3] = 0
    a = cv2.GaussianBlur(a, (3, 3), 0.5)
    col = f.astype(np.float32)
    part = (a > 0.01) & (a < 0.99)
    col[part] = np.clip(col[part] / a[part][:, None], 0, 255)          # un-premultiply (black background)
    return col, a

def stacked(i):
    col, a = key(frames[i])
    cw = cv2.warpAffine(col, B, (VW, VH), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    aw = cv2.warpAffine(a, B, (VW, VH), flags=cv2.INTER_LINEAR, borderValue=0)
    cw = np.clip(cw, 0, 255).astype(np.uint8); aw = np.clip(aw * 255, 0, 255).astype(np.uint8)
    return np.vstack([cw, cv2.cvtColor(aw, cv2.COLOR_GRAY2BGR)]), cw, aw

def encode(args, name):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{VW}x{2 * VH}",
                          "-r", str(fps), "-i", "-"] + args + [str(OUT / name)], stdin=subprocess.PIPE)
    for i in range(len(frames)):
        st, _, _ = stacked(i)
        p.stdin.write(st.tobytes())
    p.stdin.close(); p.wait()
    print(name, (OUT / name).stat().st_size // 1024, "kB")

if __name__ == "__main__":
    st, cw, aw = stacked(0)
    cv2.imwrite(str(OUT / "frame0_rgba.png"), np.dstack([cw, aw]))
    st2, cw2, aw2 = stacked(len(frames) // 2)
    cv2.imwrite(str(OUT / "frame_mid_rgba.png"), np.dstack([cw2, aw2]))
    json.dump({"k": k, "O": O.tolist(), "size": [VW, VH], "fps": fps, "frames": len(frames)}, open(OUT / "overlay.json", "w"), indent=1)
    encode(["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-movflags", "+faststart", "-an"], "dyn_anim.mp4")
    encode(["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-deadline", "good", "-cpu-used", "2", "-row-mt", "1",
            "-pix_fmt", "yuv420p", "-an"], "dyn_anim.webm")
