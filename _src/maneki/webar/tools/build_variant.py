"""Turn a reference-style A4 print (cat drawn at the reference position) into
the AR scene: writes layout json for build_scene.py and runs it.

  python3 build_variant.py <print_png>
"""
import json, subprocess, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
png = Path(sys.argv[1]).resolve()
cat = cv2.imread(str(ROOT / "animation/out/cat_master.png"), cv2.IMREAD_UNCHANGED)
ox, oy = map(int, open(ROOT / "animation/out/cat_master_offset.txt").read().split())
S, X, Y = 0.78797, 295.3 + (float(sys.argv[2]) if len(sys.argv) > 2 else 0.0), 191.8   # video frame -> A4 print; optional horizontal shift
lay = {"cat": {"h": cat.shape[0] * S, "x": X + S * ox, "y": Y + S * oy}}
lj = png.with_suffix(".layout.json")
json.dump(lay, open(lj, "w"), indent=1)
subprocess.run([sys.executable, str(ROOT / "webar/tools/build_scene.py"), str(png), str(lj),
                str(ROOT / "animation/out/cat_master.png"), str(ROOT / "animation/out/cat_master_offset.txt")], check=True)
print("ok", lay)
