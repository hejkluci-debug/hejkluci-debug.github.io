"""Build a 2-layer rig of the Maneki-neko from the reference artwork.

Outputs (in animation/out/):
  cat_full.png   - original cat with alpha (cropped to its bounding box)
  cat_body.png   - cat with the raised arm removed; the torso hidden behind
                   the arm is reconstructed by inpainting
  cat_arm.png    - the raised arm only (full canvas size, alpha)
  rig.json       - pivot (shoulder), arm axis and skinning parameters

Coordinates in rig.json are in pixels of the cropped canvas.
The raised paw is the one on the viewer's right side of the artwork.
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../assets/ref/cat_rgba_full.png")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

rgba = cv2.imread(str(SRC), cv2.IMREAD_UNCHANGED)
H, W = rgba.shape[:2]

# --- geometry measured on the 1954x2764 source (see docs/rig-notes.md) ---
ARM_POLY = np.array([
    (1250, 760), (1250, 1330), (1268, 1338), (1268, 1690), (1300, 1702),
    (1380, 1728), (1480, 1704), (1600, 1624), (1790, 1600), (1790, 760)], np.int32)
# torso surface that is hidden behind the arm and must exist in the body layer
TORSO_BEHIND = np.array([
    (1240, 1330), (1300, 1400), (1345, 1540), (1385, 1700), (1395, 1760),
    (1240, 1760)], np.int32)
PIVOT = (1400.0, 1610.0)          # shoulder joint
PAW = (1560.0, 900.0)             # centre of the paw, defines the arm axis
RAMP = (-60.0, 120.0)             # skinning ramp along the axis (px from pivot)

alpha = rgba[..., 3]
arm_region = np.zeros((H, W), np.uint8)
cv2.fillPoly(arm_region, [ARM_POLY], 255)
# above the collar the head and the paw are separated by a transparent gap:
# cut the arm region at the middle of that gap, row by row, so no head pixels
# travel with the arm
solid = alpha > 128
last = None
for y in range(740, 1345):
    xs_gap = np.where(~solid[y, 1260:1470])[0]
    if len(xs_gap):
        # first transparent run right of the head
        start = xs_gap[0]; end = start
        while end + 1 < 210 and not solid[y, 1260 + end + 1]:
            end += 1
        last = 1260 + (start + end) // 2
    if last is not None:
        arm_region[y, :last] = 0
arm_mask = cv2.bitwise_and(arm_region, alpha)

# ---------- arm layer ----------
arm = rgba.copy()
arm[..., 3] = arm_mask

# ---------- body layer ----------
body = rgba.copy()
torso = np.zeros((H, W), np.uint8)
cv2.fillPoly(torso, [TORSO_BEHIND], 255)
hole = cv2.bitwise_and(arm_region, torso)                  # torso pixels to rebuild
remove = cv2.bitwise_and(arm_region, cv2.bitwise_not(torso))  # becomes transparent
# inpaint colour of the hidden torso from the surrounding white porcelain
bgr = body[..., :3].copy()
# keep only body pixels that are not arm as inpainting source
inpaint_mask = cv2.dilate(hole, np.ones((5, 5), np.uint8))
bgr_in = cv2.inpaint(bgr, inpaint_mask, 25, cv2.INPAINT_TELEA)
body[..., :3] = np.where(inpaint_mask[..., None] > 0, bgr_in, bgr)
a = body[..., 3].copy()
a[remove > 0] = 0
a[hole > 0] = 255
# keep the torso silhouette smooth: hole limited to where alpha originally was solid
body[..., 3] = a

# ---------- crop everything to a common bounding box ----------
ys, xs = np.where(alpha > 8)
pad = 120  # room for the arm to swing outwards
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 20, H)
crop = lambda im: im[y0:y1, x0:x1]
for name, im in (("cat_full", rgba), ("cat_body", body), ("cat_arm", arm)):
    cv2.imwrite(str(OUT / f"{name}.png"), crop(im))

rig = {
    "canvas": [int(x1 - x0), int(y1 - y0)],
    "source_offset": [int(x0), int(y0)],
    "pivot": [PIVOT[0] - x0, PIVOT[1] - y0],
    "paw": [PAW[0] - x0, PAW[1] - y0],
    "ramp": list(RAMP),
    "arm_bbox": [int(ARM_POLY[:, 0].min() - x0), int(ARM_POLY[:, 1].min() - y0),
                 int(ARM_POLY[:, 0].max() - x0), int(ARM_POLY[:, 1].max() - y0)],
    "motion": {"min_deg": -4.0, "max_deg": 14.0, "period_s": 1.6},
}
json.dump(rig, open(OUT / "rig.json", "w"), indent=1)
print(json.dumps(rig))
