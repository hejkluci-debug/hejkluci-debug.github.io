"""Build target.png + scene.json for the VIDEO mode of the WebAR page.

  python3 build_scene.py <layout.png> <layout.json> <cat.png> <cat_offset_txt>

layout.json must contain {"cat": {"h", "x", "y"}} = where cat.png (scaled to
height h) sits on the print canvas. cat_offset_txt holds the offset of cat.png
inside the original video frame (so the animated overlay lines up).
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "webar/public/assets"
lay_png, lay_json, cat_png, off_txt = sys.argv[1:5]
canvas = cv2.imread(lay_png)
lay = json.load(open(lay_json))
cat = cv2.imread(cat_png, cv2.IMREAD_UNCHANGED)
ox, oy = map(int, open(off_txt).read().split())
vid = json.load(open(OUT / "video_overlay.json"))

ink = (canvas < 245).any(axis=2)
ys, xs = np.where(ink)
m = 30
tx0, ty0 = max(xs.min() - m, 0), max(ys.min() - m, 0)
tx1, ty1 = min(xs.max() + m, canvas.shape[1]), min(ys.max() + m, canvas.shape[0])
tw, th = tx1 - tx0, ty1 - ty0
target = cv2.resize(canvas[ty0:ty1, tx0:tx1], (900, int(900 * th / tw)), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(ROOT / "webar/tools/target.png"), target)

s = lay["cat"]["h"] / cat.shape[0]
cx, cy = lay["cat"]["x"], lay["cat"]["y"]
affine = {"sx": s / tw, "tx": (cx - tx0) / tw - 0.5, "sy": -s / tw, "ty": -((cy - ty0) / tw - th / tw / 2)}
X0, Y0, X1, Y1 = vid["video_px_rect"]
scene = {
    "mode": "video",
    "target": {"px": [int(tw), int(th)], "aspect": th / tw},
    "affine": affine,
    "video": {"rect": [X0 - ox, Y0 - oy, X1 - ox, Y1 - oy], "src": ["cat_anim.mp4", "cat_anim.webm"],
              "size": vid["size"]},
}
json.dump(scene, open(OUT / "scene.json", "w"), indent=1)
print(json.dumps(scene))
