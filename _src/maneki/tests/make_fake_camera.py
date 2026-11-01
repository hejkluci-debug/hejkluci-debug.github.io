"""Fake camera feed for Chromium (--use-file-for-fake-video-capture):
a T-shirt-like fabric with the printed artwork, handheld motion, perspective,
dimmer light and sensor noise. First 1.5 s show only fabric (no target).
Output: tests/fake_camera.y4m (480x640, 15 fps, 12 s)."""
import math, sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
art = cv2.imread(str(ROOT / "webar/tools/target.png"))
W, H, FPS, SECS = 480, 640, 15, 12
rng = np.random.default_rng(7)
fabric = np.clip(232 + rng.normal(0, 6, (H * 2, W * 2)), 0, 255).astype(np.uint8)
fabric = cv2.GaussianBlur(fabric, (3, 3), 0)
fabric = cv2.cvtColor(fabric, cv2.COLOR_GRAY2BGR)
# print the artwork onto fabric (multiply blend, slightly washed out like DTF on cotton)
ah = int(W * 1.3 * art.shape[0] / art.shape[1]); aw = int(W * 1.3)
a = cv2.resize(art, (aw, ah), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
y0, x0 = (fabric.shape[0] - ah) // 2, (fabric.shape[1] - aw) // 2
roi = fabric[y0:y0 + ah, x0:x0 + aw].astype(np.float32)
fabric[y0:y0 + ah, x0:x0 + aw] = np.clip(roi * (0.12 + 0.88 * a), 0, 255).astype(np.uint8)
corners = np.float32([[x0, y0], [x0 + aw, y0], [x0 + aw, y0 + ah], [x0, y0 + ah]])
out = open(ROOT / "tests/fake_camera.y4m", "wb")
out.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
for i in range(FPS * SECS):
    t = i / FPS
    if t < 1.5:   # camera pointed at plain fabric
        frame = cv2.resize(fabric[:H, :W], (W, H))
    else:
        tt = t - 1.5
        if 3.0 <= t <= 9.0: tt = 1.5   # hold the phone still -> registration check
        s = 0.42 + 0.03 * math.sin(tt * 0.9)          # distance
        ang = math.radians(4 * math.sin(tt * 1.3))     # roll
        tilt = 0.06 * math.sin(tt * 0.7)               # perspective
        cx, cy = W / 2 + 12 * math.sin(tt * 1.7), H / 2 + 10 * math.cos(tt * 1.1)
        c = corners.mean(0)
        dst = []
        for k, (px, py) in enumerate(corners):
            dx, dy = (px - c[0]) * s, (py - c[1]) * s
            dx *= 1 + (tilt if k in (1, 2) else -tilt)
            rx = dx * math.cos(ang) - dy * math.sin(ang); ry = dx * math.sin(ang) + dy * math.cos(ang)
            dst.append([cx + rx, cy + ry])
        M = cv2.getPerspectiveTransform(corners, np.float32(dst))
        frame = cv2.warpPerspective(fabric, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    light = 0.8 + 0.08 * math.sin(t * 0.5)
    frame = np.clip(frame.astype(np.float32) * light + rng.normal(0, 4, frame.shape), 0, 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0.8)
    out.write(b"FRAME\n"); out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).tobytes())
    if i == FPS * 4: cv2.imwrite(str(ROOT / "tests/fake_camera_sample.png"), frame)
out.close()
print("ok")
