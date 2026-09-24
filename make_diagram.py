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

    # ── Compute axes ──
    pts2 = np.array(ss.exterior.coords[:-1], dtype=np.float32)
    rect = cv2.minAreaRect(pts2)
    (rcx, rcy), (rw, rh), rang = rect
    major_angle = rang      if rw >= rh else rang + 90
    minor_angle = major_angle + 90

    cen  = centroid_np(ss)
    pcen = centroid_np(ps)

    # Unit vectors
    major_vec = np.array([math.cos(math.radians(major_angle)),
                           math.sin(math.radians(major_angle))])
    minor_vec = np.array([math.cos(math.radians(minor_angle)),
                           math.sin(math.radians(minor_angle))])

    # Chord endpoints
    maj_p1, maj_p2 = intersection_endpoints(ss, cen,  major_angle)
    min_p1, min_p2 = intersection_endpoints(ss, cen,  minor_angle)
    por_p1, por_p2 = intersection_endpoints(ps, pcen, minor_angle)

    if maj_p1 is None or min_p1 is None or por_p1 is None:
        print("WARNING: some chord computation failed")

    # ── Separate the three "minor direction" lines by offsetting along major_vec ──
    # Minor axis (stoma): no offset
    # Pore width:         +major_offset px along major axis direction
    # Guard cell bracket: same offset as pore

    MAJOR_OFF = major_vec * 35   # pixels to shift right along major axis

    # ── Draw Major axis (no offset) ──
    if maj_p1 is not None:
        double_arrow(crop, maj_p1, maj_p2, C_MAJOR, 3)
        # label above the midpoint
        mid = (maj_p1+maj_p2)/2
        put_text(crop, "Major axis", mid + np.array([-50, -22]), C_MAJOR)

    # ── Draw Minor axis (stoma width) ──
    if min_p1 is not None:
        double_arrow(crop, min_p1, min_p2, C_MINOR, 3)
        mid = (min_p1+min_p2)/2
        put_text(crop, "Minor axis", mid + np.array([10, 25]), C_MINOR)

    # ── Draw Pore width (offset along major direction) ──
    if por_p1 is not None:
        pp1 = por_p1 + MAJOR_OFF
        pp2 = por_p2 + MAJOR_OFF
        double_arrow(crop, pp1, pp2, C_PORE_LINE, 3)
        mid = (pp1+pp2)/2
        put_text(crop, "Pore width", mid + np.array([10, 25]), C_PORE_LINE)

        # ── Guard cell (one side: from pore-edge to stoma-edge, same offset) ──
        if min_p1 is not None:
            mdir   = (min_p2-min_p1) / np.linalg.norm(min_p2-min_p1)
            d1     = np.dot(por_p1 - cen, mdir)
            d2     = np.dot(por_p2 - cen, mdir)
            por_near  = por_p1 if d1 < d2 else por_p2   # pore end toward negative direction
            sto_near  = min_p1 if np.dot(min_p1-cen, mdir) < 0 else min_p2  # stoma end same side

            gc_p1 = por_near  + MAJOR_OFF
            gc_p2 = sto_near  + MAJOR_OFF
            draw_off = major_vec * (-35)  # guard cell on opposite side
            double_arrow(crop, por_near + draw_off, sto_near + draw_off, C_GUARD, 3)
            mid_gc = (por_near + sto_near)/2 + draw_off
            put_text(crop, "Guard cell", mid_gc + np.array([-110, 25]), C_GUARD)

    # ── Legend ──
    legend = [
        (C_STOMA,     "Stoma body  (class 1)"),
        (C_PORE_POLY, "Pore / aperture  (class 0)"),
        (C_MAJOR,     "Major axis length"),
        (C_MINOR,     "Minor axis  (stoma width)"),
        (C_PORE_LINE, "Pore width"),
        (C_GUARD,     "Guard cell = (minor \u2212 pore) / 2"),
    ]
    lx, ly, lh, lw = 10, 10, 30, 360
    ov = crop.copy()
    cv2.rectangle(ov, (lx-6, ly-6), (lx+lw, ly+len(legend)*lh+10), (15,15,15), -1)
    cv2.addWeighted(ov, 0.70, crop, 0.30, 0, crop)
    for i, (col, label) in enumerate(legend):
        y = ly + i*lh + lh//2
        cv2.rectangle(crop, (lx, y-7), (lx+20, y+7), col, -1)
        cv2.putText(crop, label, (lx+28, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE, 1, cv2.LINE_AA)

    # Save alongside measurements.qmd (picked up by `quarto render`)
    out_root = OUT_ROOT / "stomata_diagram.png"
    cv2.imwrite(str(out_root), crop)
    # Also overwrite in docs/ so the pre-built site is immediately correct
    out_docs = OUT_DOCS / "stomata_diagram.png"
    cv2.imwrite(str(out_docs), crop)
    print(f"Saved {out_root}  and  {out_docs}  ({crop.shape[1]}x{crop.shape[0]})")


if __name__ == "__main__":
    main()
