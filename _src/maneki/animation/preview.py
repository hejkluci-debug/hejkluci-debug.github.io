"""Render a preview video of the rig with exactly the same skinning math as the
web player (webar/js/cat-rig.js): every point of the arm layer is rotated
around the shoulder pivot by angle * w, where w ramps 0 -> 1 along the arm axis.
Output: animation/out/preview.mp4 and preview_strip.png (6 key poses).
"""
import json, math, subprocess
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).parent / "out"
rig = json.load(open(OUT / "rig.json"))
body = cv2.imread(str(OUT / "cat_body.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
arm = cv2.imread(str(OUT / "cat_arm.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
H, W = body.shape[:2]
px, py = rig["pivot"]
ax, ay = rig["paw"][0] - px, rig["paw"][1] - py
n = math.hypot(ax, ay); ax, ay = ax / n, ay / n
r0, r1 = rig["ramp"]
m = rig["motion"]

def smooth(t):
    t = np.clip((t - r0) / (r1 - r0), 0, 1)
    return t * t * (3 - 2 * t)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

def angle_at(t):  # same easing as the web player
    s = 0.5 - 0.5 * math.cos(2 * math.pi * t / m["period_s"])
    return math.radians(m["min_deg"] + (m["max_deg"] - m["min_deg"]) * s)

def warp_arm(theta):
    # inverse mapping: for each destination pixel find the source pixel
    dx, dy = xx - px, yy - py
    sx, sy = xx.copy(), yy.copy()
    for _ in range(4):
        w = smooth((sx - px) * ax + (sy - py) * ay)
        a = -theta * w
        c, s = np.cos(a), np.sin(a)
        sx = px + c * dx - s * dy
        sy = py + s * dx + c * dy
    return cv2.remap(arm, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def compose(theta, bg=(235, 235, 235)):
    a = warp_arm(theta)
    out = np.empty((H, W, 3), np.float32); out[:] = bg
    for layer in (body, a):
        al = layer[..., 3:4] / 255.0
        out = layer[..., :3] * al + out * (1 - al)
    return out.astype(np.uint8)

scale = 0.4
frames = []
fps = 25
for i in range(int(m["period_s"] * fps)):
    img = compose(angle_at(i / fps))
    frames.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
h, w = frames[0].shape[:2]
h -= h % 2; w -= w % 2
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                      "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vf", "loop=3:40:0",
                      "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                      str(OUT / "preview.mp4")], stdin=subprocess.PIPE)
for f in frames:
    p.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
p.stdin.close(); p.wait()
idx = np.linspace(0, len(frames) // 2, 6).astype(int)
cv2.imwrite(str(OUT / "preview_strip.png"), np.hstack([frames[i] for i in idx]))
print("frames", len(frames), "size", w, h)
