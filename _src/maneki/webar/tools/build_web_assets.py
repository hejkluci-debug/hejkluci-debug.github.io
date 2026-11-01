"""Build everything the WebAR page needs from the rig and the chosen layout.

Outputs to webar/public/assets/:
  target.png     - the printed artwork, cropped; compiled to target.mind
  arm.webp/png   - the animated arm (cropped)
  underlay.webp/png - covers the printed arm in the area the virtual arm sweeps
                   through, and shows the torso that the arm hides in print
  scene.json     - all coordinates for the player (pixel space of cat_full +
                   affine transform into MindAR anchor space)
"""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANIM = ROOT / "animation" / "out"
LAY = ROOT / "builds" / "layouts"
OUT = ROOT / "webar" / "public" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
TEX_SCALE = 0.55                      # texture resolution vs. 1954x2764 source

rig = json.load(open(ANIM / "rig.json"))
lay = json.load(open(LAY / "layout_behind.json"))
canvas = cv2.imread(str(LAY / "layout_behind.png"))
full = cv2.imread(str(ANIM / "cat_full.png"), cv2.IMREAD_UNCHANGED)
body = cv2.imread(str(ANIM / "cat_body.png"), cv2.IMREAD_UNCHANGED)
arm = cv2.imread(str(ANIM / "cat_arm.png"), cv2.IMREAD_UNCHANGED)
H, W = full.shape[:2]

# ---- target image: printed artwork cropped to its content ----
ink = (canvas < 245).any(axis=2)
ys, xs = np.where(ink)
m = 30
tx0, ty0 = max(xs.min() - m, 0), max(ys.min() - m, 0)
tx1, ty1 = min(xs.max() + m, canvas.shape[1]), min(ys.max() + m, canvas.shape[0])
target = canvas[ty0:ty1, tx0:tx1]
tw, th = tx1 - tx0, ty1 - ty0
tsmall = cv2.resize(target, (900, int(900 * th / tw)), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(ROOT / "webar" / "tools" / "target.png"), tsmall)

# cat_full pixel -> canvas pixel: X = cx + px*s ; then -> anchor units
s = lay["cat"]["h"] / H
cx, cy = lay["cat"]["x"], lay["cat"]["y"]
# anchor: width 1, origin centre, y up
# u = (cx + px*s - tx0)/tw - 0.5 ; v = -((cy + py*s - ty0)/tw - th/tw/2)
affine = {"sx": s / tw, "tx": (cx - tx0) / tw - 0.5,
          "sy": -s / tw, "ty": -((cy - ty0) / tw - th / tw / 2)}

# ---- sweep region of the arm over the whole motion ----
px, py = rig["pivot"]; ax, ay = rig["paw"][0] - px, rig["paw"][1] - py
n = math.hypot(ax, ay); ax, ay = ax / n, ay / n
r0, r1 = rig["ramp"]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def smooth(t):
    t = np.clip((t - r0) / (r1 - r0), 0, 1); return t * t * (3 - 2 * t)
def warp(theta):
    dx, dy = xx - px, yy - py; sx, sy = xx.copy(), yy.copy()
    for _ in range(4):
        w = smooth((sx - px) * ax + (sy - py) * ay); a = -theta * w
        c, si = np.cos(a), np.sin(a); sx = px + c * dx - si * dy; sy = py + si * dx + c * dy
    return cv2.remap(arm[..., 3], sx, sy, cv2.INTER_LINEAR)
sweep = arm[..., 3].copy()
for deg in np.linspace(rig["motion"]["min_deg"] - 1, rig["motion"]["max_deg"] + 1, 25):
    sweep = np.maximum(sweep, warp(math.radians(deg)))
sweep = cv2.dilate(sweep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
feather = cv2.GaussianBlur(sweep.astype(np.float32), (0, 0), 10) / 255.0

# underlay = shirt-white + reconstructed torso, only inside the sweep region
under = np.zeros((H, W, 4), np.float32)
under[..., :3] = 255
ba = body[..., 3:4] / 255.0
under[..., :3] = body[..., :3] * ba + under[..., :3] * (1 - ba)
under[..., 3] = 255 * np.clip(feather * 1.15, 0, 1)
under = under.astype(np.uint8)

def crop_save(img, name):
    a = img[..., 3]; ys, xs = np.where(a > 2)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    c = img[y0:y1, x0:x1]
    c = cv2.resize(c, None, fx=TEX_SCALE, fy=TEX_SCALE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{name}.png"), c, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(OUT / f"{name}.webp"), c, [cv2.IMWRITE_WEBP_QUALITY, 88])
    return [x0, y0, x1, y1]

scene = {
    "target": {"px": [int(tw), int(th)], "aspect": th / tw},
    "affine": affine,
    "rig": rig,
    "arm_rect": crop_save(arm, "arm"),
    "underlay_rect": crop_save(under, "underlay"),
    "mesh": {"cols": 18, "rows": 32},
}
json.dump(scene, open(OUT / "scene.json", "w"), indent=1)
for f in sorted(OUT.iterdir()):
    print(f.name, f.stat().st_size)
