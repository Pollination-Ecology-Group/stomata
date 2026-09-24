"""
make_diagram.py -- Stomata measurement documentation figure.
Strategy:
  - Major axis: drawn through stoma centroid (long axis of guard cells)
  - Minor axis: drawn through stoma centroid, perpendicular
  - Pore width: drawn through PORE centroid (shorter line on pore polygon)
  - Guard cell: bracket on the outside of the stoma (half of minor - pore)

All three perpendicular lines are offset from each other to prevent overlap.
Output: docs/images/stomata_diagram.png
"""

import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, LineString
import math

DATASET    = Path("/home/meow/cesnet_data/Ostatni/Stomata/datasets/stomata_v1/train")
IMAGE_FILE = DATASET / "images" / "image008_train0_png.rf.5bfed52f2692fc47ce90b6a294d5438d.jpg"
LABEL_FILE = DATASET / "labels" / "image008_train0_png.rf.5bfed52f2692fc47ce90b6a294d5438d.txt"
# Write alongside measurements.qmd (Quarto will copy to docs/) AND directly to docs/
OUT_ROOT = Path(".")      # alongside measurements.qmd
OUT_DOCS = Path("docs")   # pre-built output dir

# Colours (BGR)
C_STOMA     = ( 80, 210,  80)
C_PORE_POLY = ( 60, 200, 240)
C_MAJOR     = ( 20, 235, 255)
C_MINOR     = ( 30,  90, 255)
C_PORE_LINE = (200,  80,  50)
C_GUARD     = (230,  40, 230)
C_WHITE     = (255, 255, 255)


def parse_labels(label_path, img_w, img_h):
    result = {0: [], 1: []}
    for line in label_path.read_text().strip().splitlines():
        parts = list(map(float, line.split()))
        cls   = int(parts[0])
        xs    = [c * img_w for c in parts[1::2]]
        ys    = [c * img_h for c in parts[2::2]]
        pts   = list(zip(xs, ys))
        if len(pts) >= 3:
            result[cls].append(Polygon(pts))
    return result


def centroid_np(poly):
    return np.array([poly.centroid.x, poly.centroid.y])


def intersection_endpoints(poly, through_pt, angle_deg, length=5000):
    """Find the two boundary intersection points of a line through through_pt."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad)*length, math.sin(rad)*length
    cx, cy = through_pt
    line  = LineString([(cx-dx, cy-dy), (cx+dx, cy+dy)])
    inter = poly.boundary.intersection(line)
    if inter.is_empty:
        inter = poly.intersection(line)
    if inter.is_empty:
        return None, None
    pts = []
    if hasattr(inter, 'geoms'):
        for g in inter.geoms:
            if hasattr(g, 'x'):
                pts.append(np.array([g.x, g.y]))
            elif hasattr(g, 'coords'):
                pts.extend(np.array(c) for c in g.coords)
    elif hasattr(inter, 'coords'):
        pts = [np.array(c) for c in inter.coords]
    if len(pts) < 2:
        return None, None
    # Return the two most-distant points
    max_d, p1, p2 = 0, pts[0], pts[1]
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i]-pts[j])
            if d > max_d:
                max_d, p1, p2 = d, pts[i], pts[j]
    return p1, p2


def double_arrow(img, p1, p2, col, thick=3):
    p1i = tuple(np.round(p1).astype(int))
    p2i = tuple(np.round(p2).astype(int))
    L   = max(np.linalg.norm(p2-p1), 1)
    tip = min(0.13, 15/L)
    cv2.arrowedLine(img, p1i, p2i, col, thick, tipLength=tip, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, p2i, p1i, col, thick, tipLength=tip, line_type=cv2.LINE_AA)


def fill_poly(img, poly, col, alpha=0.18, border=2):
    pts = np.array(poly.exterior.coords[:-1], dtype=np.int32).reshape((-1,1,2))
    ov  = img.copy()
    cv2.fillPoly(ov, [pts], col)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.polylines(img, [pts], True, col, border, lineType=cv2.LINE_AA)


def put_text(img, s, pos, col, fscale=0.75, thick=2):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, s, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, s, (x,   y  ), cv2.FONT_HERSHEY_SIMPLEX, fscale,       col, thick,   cv2.LINE_AA)


def leader_line(img, arrow_mid, label_pos, col):
    """Thin line from arrow midpoint to label."""
    p1 = tuple(np.round(arrow_mid).astype(int))
    p2 = tuple(np.round(label_pos).astype(int))
    cv2.line(img, p1, p2, col, 1, lineType=cv2.LINE_AA)


def main():
    img  = cv2.imread(str(IMAGE_FILE))
    assert img is not None
    H, W = img.shape[:2]

    lbl     = parse_labels(LABEL_FILE, W, H)
    stomata = lbl[1]; pores = lbl[0]

    # Largest stoma + matched pore
    best_s, best_p, best_a = None, None, 0
    for s in stomata:
        for p in pores:
            if s.contains(p.centroid) or s.intersects(p):
                if s.area > best_a:
                    best_s, best_p, best_a = s, p, s.area
    if best_s is None:
        best_s, best_p = stomata[0], pores[0]

    # Crop
    minx, miny, maxx, maxy = best_s.bounds
    ext = max(maxx-minx, maxy-miny) * 0.85
    x0, y0 = max(0, int(minx-ext)), max(0, int(miny-ext))
    x1, y1 = min(W, int(maxx+ext)), min(H, int(maxy+ext))
    crop = img[y0:y1, x0:x1].copy()

    def tran(p, dx, dy): return Polygon([(x-dx, y-dy) for x,y in p.exterior.coords])
    def scal(p, s):      return Polygon([(x*s, y*s)   for x,y in p.exterior.coords])

    S    = 3
    crop = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_LANCZOS4)
    ss   = scal(tran(best_s, x0, y0), S)
    ps   = scal(tran(best_p, x0, y0), S)

    # CLAHE
    lab    = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l,a,b  = cv2.split(lab)
    l      = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(l)
    crop   = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

    fill_poly(crop, ss, C_STOMA,     alpha=0.17, border=3)
    fill_poly(crop, ps, C_PORE_POLY, alpha=0.32, border=2)

    # ── Compute axes from the PORE polygon ──
    # The pore (class 0) is elongated — its long axis defines the stoma orientation.
    # All four measurements are derived from this direction.
    pore_pts = np.array(ps.exterior.coords[:-1], dtype=np.float32)
    pore_rect = cv2.minAreaRect(pore_pts)
    (prcx, prcy), (prw, prh), prang = pore_rect
    # major_angle = direction of pore's long axis
    major_angle = prang      if prw >= prh else prang + 90
    minor_angle = major_angle + 90   # perpendicular = stoma width direction

    cen  = centroid_np(ss)
    pcen = centroid_np(ps)

    # Unit vectors
    major_vec = np.array([math.cos(math.radians(major_angle)),
                           math.sin(math.radians(major_angle))])
    minor_vec = np.array([math.cos(math.radians(minor_angle)),
                           math.sin(math.radians(minor_angle))])

    # Chord endpoints
    # Major: through stoma centroid along pore's long axis → full stoma length
    maj_p1, maj_p2 = intersection_endpoints(ss, cen,  major_angle)
    # Minor: through stoma centroid perpendicular → full stoma width
    min_p1, min_p2 = intersection_endpoints(ss, cen,  minor_angle)
    # Pore width: through pore centroid along minor_angle → pore aperture width
    por_p1, por_p2 = intersection_endpoints(ps, pcen, minor_angle)

    if maj_p1 is None or min_p1 is None or por_p1 is None:
        print("WARNING: some chord computation failed")

    # ── Separate the three "minor direction" lines by offsetting along major_vec ──
    # Minor axis (stoma): no offset
    # Pore width:         +major_offset px along major axis direction
    # Guard cell bracket: same offset as pore

    MAJOR_OFF = major_vec * 35   # shift pore/guard arrows perpendicular

    # ── Draw Major axis (through stoma centroid, vertical = pore long axis) ──
    if maj_p1 is not None:
        double_arrow(crop, maj_p1, maj_p2, C_MAJOR, 3)
        # label at whichever tip is higher on screen (smaller y)
        tip = maj_p1 if maj_p1[1] < maj_p2[1] else maj_p2
        put_text(crop, "Major axis", tip + np.array([8, -10]), C_MAJOR)

    # ── Draw Minor axis (horizontal, through stoma centroid) ──
    if min_p1 is not None:
        double_arrow(crop, min_p1, min_p2, C_MINOR, 3)
        # label at the right-most arrowhead
        tip = min_p1 if min_p1[0] > min_p2[0] else min_p2
        put_text(crop, "Minor axis", tip + np.array([6, -8]), C_MINOR)

    # ── Draw Pore width (offset +35px along major_vec from stoma centre) ──
    if por_p1 is not None:
        pp1 = por_p1 + MAJOR_OFF
        pp2 = por_p2 + MAJOR_OFF
        double_arrow(crop, pp1, pp2, C_PORE_LINE, 3)
        # label at the right-most tip
        tip = pp1 if pp1[0] > pp2[0] else pp2
        put_text(crop, "Pore width", tip + np.array([6, -8]), C_PORE_LINE)

        # ── Guard cell (opposite side, -35px along major_vec) ──
        if min_p1 is not None:
            mdir      = (min_p2-min_p1) / np.linalg.norm(min_p2-min_p1)
            d1        = np.dot(por_p1 - cen, mdir)
            d2        = np.dot(por_p2 - cen, mdir)
            por_near  = por_p1 if d1 < d2 else por_p2
            sto_near  = min_p1 if np.dot(min_p1-cen, mdir) < 0 else min_p2
            draw_off  = major_vec * (-35)
            gc_a = por_near + draw_off
            gc_b = sto_near + draw_off
            double_arrow(crop, gc_a, gc_b, C_GUARD, 3)
            # label at the left-most tip
            tip = gc_a if gc_a[0] < gc_b[0] else gc_b
            put_text(crop, "Guard cell", tip + np.array([-115, -8]), C_GUARD)

    # ── Legend — rendered as a separate strip BELOW the image ──
    legend = [
        (C_STOMA,     "Stoma body (class 1)"),
        (C_PORE_POLY, "Pore / aperture (class 0)"),
        (C_MAJOR,     "Major axis length"),
        (C_MINOR,     "Minor axis  (stoma width)"),
        (C_PORE_LINE, "Pore width"),
        (C_GUARD,     "Guard cell width = (minor \u2212 pore) / 2"),
    ]
    IW = crop.shape[1]
    LH = 32           # row height in legend strip
    COLS = 2          # two-column layout
    ROWS = math.ceil(len(legend) / COLS)
    STRIP_H = ROWS * LH + 16
    strip = np.full((STRIP_H, IW, 3), 28, dtype=np.uint8)  # dark background

    col_w = IW // COLS
    for i, (col, label) in enumerate(legend):
        row = i % ROWS
        c   = i // ROWS
        x   = c * col_w + 10
        y   = row * LH + LH // 2 + 8
        cv2.rectangle(strip, (x, y-8), (x+18, y+8), col, -1)
        cv2.putText(strip, label, (x+24, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1, cv2.LINE_AA)

    # Thin separator line
    sep = np.full((3, IW, 3), 80, dtype=np.uint8)
    final = np.vstack([crop, sep, strip])

    # Save alongside measurements.qmd (picked up by `quarto render`)
    out_root = OUT_ROOT / "stomata_diagram.png"
    cv2.imwrite(str(out_root), final)
    # Also overwrite in docs/ so the pre-built site is immediately correct
    out_docs = OUT_DOCS / "stomata_diagram.png"
    cv2.imwrite(str(out_docs), final)
    print(f"Saved {out_root}  and  {out_docs}  ({final.shape[1]}x{final.shape[0]})")


if __name__ == "__main__":
    main()
