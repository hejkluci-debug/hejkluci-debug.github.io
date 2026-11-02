"""Export print deliverables from an A4 300 dpi master PNG.

  python3 export_print.py <master_png> <out_dir> <prefix>

Writes <prefix>_testovaci_arch_A4.pdf, <prefix>_tisk_A4_300dpi_bile_pozadi.png,
<prefix>_tisk_A4_300dpi_pruhledne_jen_bile_triko.png (pure white -> transparent)
and nahled_<prefix>.jpg.
"""
import sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image

src, out, pre = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
bgr = cv2.imread(src, cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
im = Image.fromarray(rgb)
im.save(out / f"{pre}_testovaci_arch_A4.pdf", "PDF", resolution=300, quality=95)
cv2.imwrite(str(out / f"{pre}_tisk_A4_300dpi_bile_pozadi.png"), bgr)
alpha = np.where(bgr.min(2) >= 250, 0, 255).astype(np.uint8)
cv2.imwrite(str(out / f"{pre}_tisk_A4_300dpi_pruhledne_jen_bile_triko.png"), np.dstack([bgr, alpha]))
s = 640 / bgr.shape[1]
cv2.imwrite(str(out / f"nahled_{pre}.jpg"), cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print("ok", out)
