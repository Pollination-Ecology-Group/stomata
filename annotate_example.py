"""
make_doc_figure.py  — final version
Uses a manually-identified clear stoma from 6-10-A_AB_1.png.
Produces:
  example_field.png     — CLAHE-enhanced full field view
  example_annotated.png — tightly cropped stoma with measurement arrows
"""
import cv2
import numpy as np
from pathlib import Path

DETECT_DIR = Path("/home/meow/cesnet_data/Ostatni/Stomata/detect/"
                  "stomata_trait_comparison_AA012-20251216T171633Z-3-001/"
                  "stomata_trait_comparison_AA012")

SRC = DETECT_DIR / "6-10-A_AB_1.png"
OUT_FULL = Path("/home/meow/Documents/Antigravity/Stomata/example_field.png")
OUT_ANNO = Path("/home/meow/Documents/Antigravity/Stomata/example_annotated.png")

# ── Load & enhance ────────────────────────────────────────────────────────────
img = cv2.imread(str(SRC))
assert img is not None, f"Cannot open {SRC}"
h, w = img.shape[:2]
print(f"Image: {w}×{h}")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b_ch = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
enhanced = cv2.merge([clahe.apply(l), a, b_ch])
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
cv2.imwrite(str(OUT_FULL), enhanced)
print(f"  → field view saved")

# ── Auto-find the clearest, most isolated stoma ───────────────────────────────
gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=100,
                            param1=55, param2=20, minRadius=20, maxRadius=70)
assert circles is not None, "No circles found"
circles = np.round(circles[0]).astype(int)

# Score each candidate: prefer (a) near image centre, (b) far from other circles
def isolation(c, all_c):
    dists = [np.linalg.norm(c[:2] - o[:2]) for o in all_c if not np.array_equal(c, o)]
    return min(dists) if dists else 0

margin = 350
candidates = [c for c in circles
              if margin < c[0] < w-margin and margin < c[1] < h-margin]
# pick one with good isolation and close to centre
candidates.sort(key=lambda c: -isolation(c, candidates))   # most isolated first
# among top-20 isolated, pick closest to centre
top = candidates[:20]
top.sort(key=lambda c: abs(c[0]-w//2) + abs(c[1]-h//2))
CX, CY, R = top[0]
print(f"  Stoma at ({CX}, {CY}), r={R}")

# ── Tight crop ────────────────────────────────────────────────────────────────
HALF = R * 5
x1, y1 = max(CX-HALF, 0), max(CY-HALF, 0)
x2, y2 = min(CX+HALF, w), min(CY+HALF, h)
crop = enhanced[y1:y2, x1:x2].copy()
ch, cw = crop.shape[:2]
cx, cy = CX-x1, CY-y1

# ── Estimate stoma orientation from local gradient ─────────────────────────────
roi_gray = gray[max(CY-R,0):CY+R, max(CX-R,0):CX+R]
gx = cv2.Sobel(roi_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=7)
gy = cv2.Sobel(roi_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=7)
mag = np.sqrt(gx**2 + gy**2)
mask = mag > np.percentile(mag, 85)
ang = np.arctan2(gy[mask], gx[mask])
# Major axis is perpendicular to the dominant gradient
hist, edges = np.histogram(ang, bins=180, range=(-np.pi, np.pi))
dom = edges[np.argmax(hist)] + np.pi/180
ANGLE_DEG = float(np.degrees(dom) + 90) % 180 - 90
print(f"  Stoma angle: {ANGLE_DEG:.1f}°")

angle = np.deg2rad(ANGLE_DEG)
cos_a, sin_a = np.cos(angle), np.sin(angle)

# ── Proportional geometry (all relative to R) ─────────────────────────────────
MAJOR_HALF = int(R * 2.7)
MINOR_HALF = int(R * 1.7)
PORE_HALF  = int(R * 0.5)
GUARD_W    = MINOR_HALF - PORE_HALF

def rot(dx, dy):
    return (int(cx + dx*cos_a - dy*sin_a),
            int(cy + dx*sin_a + dy*cos_a))

maj1, maj2 = rot(-MAJOR_HALF, 0), rot(MAJOR_HALF, 0)
min1, min2 = rot(0, -MINOR_HALF), rot(0, MINOR_HALF)
por1, por2 = rot(0, -PORE_HALF),  rot(0, PORE_HALF)
gc1,  gc2  = rot(0,  PORE_HALF),  rot(0, PORE_HALF+GUARD_W)

# ── Colours ───────────────────────────────────────────────────────────────────
COL_MAJ = (210,  70,  10)   # deep orange — major axis
COL_MIN = ( 20, 150,  30)   # green       — minor axis
COL_POR = ( 30,  30, 200)   # blue        — pore
COL_GC  = (160,  10, 170)   # purple      — guard cell
WHITE   = (255, 255, 255)

TK = max(4, R//14)
FS = max(0.85, R/55)

def double_arrow(im, p1, p2, col, tk=TK):
    L = max(np.linalg.norm(np.array(p2)-np.array(p1)), 1)
    tip = 14.0/L
    cv2.arrowedLine(im, p2, p1, col, tk, tipLength=tip)
    cv2.arrowedLine(im, p1, p2, col, tk, tipLength=tip)

def label(im, txt, pos, col, fs=FS, tk=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th), bl = cv2.getTextSize(txt, font, fs, tk)
    x, y = pos
    cv2.rectangle(im, (x-4, y-th-4), (x+tw+4, y+bl+2), WHITE, -1)
    cv2.putText(im, txt, (x, y), font, fs, col, tk, cv2.LINE_AA)

# Upscale 2× for crispness
SF = 2
big = cv2.resize(crop, (cw*SF, ch*SF), interpolation=cv2.INTER_CUBIC)
def s(pt): return (pt[0]*SF, pt[1]*SF)

# Draw arrows
double_arrow(big, s(maj1), s(maj2), COL_MAJ, TK*SF)
double_arrow(big, s(min1), s(min2), COL_MIN, TK*SF)
double_arrow(big, s(por1), s(por2), COL_POR, TK*SF)

# Guard cell bracket
cv2.line(big, s(gc1), s(gc2), COL_GC, TK*SF)
for pt in [gc1, gc2]:
    p = s(pt); tick = 10*SF
    cv2.line(big, (p[0]-tick, p[1]), (p[0]+tick, p[1]), COL_GC, TK*SF)

# Labels — placed outside the arrows
pad = 18*SF
label(big, "Major axis",       (s(maj2)[0]+pad,  s(maj2)[1]),           COL_MAJ, FS*SF)
label(big, "Minor axis",       (s(min2)[0]+pad,  s(min2)[1]+pad//2),   COL_MIN, FS*SF)
label(big, "Pore width",       (s(por1)[0]+pad,  s(por1)[1]-pad//2),   COL_POR, FS*SF)
label(big, "Guard cell width", (s(gc2)[0]+pad,   s(gc2)[1]+pad//2),    COL_GC,  FS*SF)

cv2.imwrite(str(OUT_ANNO), big)
print(f"  → annotated crop saved: {big.shape[1]}×{big.shape[0]}")
print("Done!")
