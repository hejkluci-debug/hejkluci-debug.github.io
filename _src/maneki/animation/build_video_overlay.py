"""Convert the user's maneki_neko_alpha.webm (VP9 + alpha, 2480x3508, 8 s)
into a light "stacked alpha" overlay that every phone can play:

  top half    = colour,  bottom half = alpha (as grey)
  H.264 MP4 (iOS Safari, Android) + VP9 WebM fallback, no audio, 2 s loop.

Findings this script relies on (measured, see docs/VIDEO_ANALYZA.md):
  * the animation loops every 60 frames (frame 30 and 60 == frame 0)
  * only the raised arm and the eyes move; the body is pixel-identical to the
    PDF artwork after an affine fit (mean diff 1.4 grey levels)
  * frame 0 is used as the print master, so the overlay matches the shirt.

Outputs: webar/public/assets/cat_anim.mp4, cat_anim.webm, video_overlay.json
         assets/print/maneki_print_master.png (frame 0, RGBA, 300 dpi A4)
"""
import json, subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/video/maneki_neko_alpha.webm"
OUT = ROOT / "webar/public/assets"
TMP = ROOT / "assets/video/loop"; TMP.mkdir(parents=True, exist_ok=True)
(ROOT / "assets/print").mkdir(exist_ok=True)
LOOP = 60
import os
FULL = os.environ.get("OVERLAY", "full") == "full"
if FULL:
    X0, Y0, X1, Y1 = 320, 392, 2232, 3240      # whole cat (union of alpha over the loop) + margin
else:
    X0, Y0, X1, Y1 = 672, 916, 2480, 2228      # changing region (arm + eyes) + margin
OUT_W = 720

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-c:v", "libvpx-vp9", "-i", str(SRC),
                "-frames:v", str(LOOP), "-pix_fmt", "rgba", str(TMP / "f_%03d.png")], check=True)
frames = [cv2.imread(str(TMP / f"f_{i:03d}.png"), cv2.IMREAD_UNCHANGED) for i in range(1, LOOP + 1)]
cv2.imwrite(str(ROOT / "assets/print/maneki_print_master.png"), frames[0])

crops = [f[Y0:Y1, X0:X1] for f in frames]
h, w = crops[0].shape[:2]
a0 = crops[0][..., 3]
# region where the animated content differs from the print (frame 0) at any time
diff = np.zeros((h, w), np.uint8)
for c in crops[1:]:
    d = np.abs(c.astype(np.int16) - crops[0].astype(np.int16)).max(axis=2)
    diff = np.maximum(diff, (d > 12).astype(np.uint8) * 255)
diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
# drop compression specks: keep only components bigger than 1500 px (eyes, arm)
n, lab, stats, _ = cv2.connectedComponentsWithStats(diff)
keep = np.zeros_like(diff)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 1500:
        keep[lab == i] = 255
diff = keep
patch = cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
patch = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), 12) / 255.0     # feathered 0..1

s = OUT_W / w
ow, oh = OUT_W, int(round(h * s / 2) * 2)
p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                      "-s", f"{ow}x{oh*2}", "-r", "30", "-i", "-", "-an",
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                      "-movflags", "+faststart", str(OUT / "cat_anim.mp4")], stdin=subprocess.PIPE)
stack_frames = []
for c in crops:
    a = c[..., 3:4].astype(np.float32) / 255.0
    # shirt-white under the moving parts so the printed arm never shows through
    col = c[..., :3] * a + 255 * (1 - a)
    if FULL:   # straight (not pre-blended) colour on the soft cat edge: no white halo on darker shirts
        col = np.where(a > 0.02, c[..., :3].astype(np.float32), 255.0)
    if FULL:   # whole cat from the video + shirt-white wherever the printed arm must be hidden
        alpha = np.maximum(a[..., 0], patch)
    else:      # overlay only where the animation differs from the print, feathered edge
        alpha = patch
    col = cv2.resize(col.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_AREA)
    al = cv2.resize((alpha * 255).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_AREA)
    st = np.vstack([col, cv2.cvtColor(al, cv2.COLOR_GRAY2BGR)])
    stack_frames.append(st)
    p.stdin.write(st.tobytes())
p.stdin.close(); p.wait()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / "cat_anim.mp4"), "-an",
                "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-row-mt", "1", "-deadline", "good",
                str(OUT / "cat_anim.webm")], check=True)
cv2.imwrite(str(TMP / "stack_preview.png"), np.hstack([stack_frames[0], stack_frames[15], stack_frames[45]]))
json.dump({"video_px_rect": [X0, Y0, X1, Y1], "video_canvas": [frames[0].shape[1], frames[0].shape[0]],
           "loop_frames": LOOP, "fps": 30, "size": [ow, oh]},
          open(OUT / "video_overlay.json", "w"), indent=1)
for f in ("cat_anim.mp4", "cat_anim.webm"):
    print(f, (OUT / f).stat().st_size)
