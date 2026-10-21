#!/usr/bin/env python3
"""
extract_traits.py
High-precision stomata and pore aperture trait extraction engine for leaf epidermis microscopy.

Features:
- Batched sliding-window tiled inference at native resolution (default 640x640, stride 448).
- CLAHE (Contrast-Limited Adaptive Histogram Equalization) contrast enhancement for out-of-focus recovery.
- Decoupled dual-confidence strategy (conf_stoma=0.18 for faint stoma recall, conf_center=0.38 for strict pore precision).
- Per-stoma Focus Quality Index (Modified Laplacian variance score, 0-100 scale).
- Separate objectives for Stomatal Density vs Ploidy Size Estimation:
    * Stomatal Density: All stomata (including blurred ones and edge stomata) are counted.
    * Ploidy Sizing: Strict filtering (is_suitable_for_ploidy_sizing: in-focus, fully interior, biometrically valid, non-truncated).
- Automated Image-Level Summary Table: Computes stomatal density (stomata/mm2) and median/mean guard cell sizes for ploidy comparison.
"""

import os
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon, LineString, MultiPolygon, Point
from shapely.ops import unary_union
from ultralytics import YOLO

# Standard microscope calibration constant (um per pixel)
PX_TO_UM = 50.0 / 451.3958


def apply_clahe_enhancement(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE on the L-channel of LAB color space to enhance subtle gradients
    in out-of-focus or low-contrast leaf microscopy regions without altering colors.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def compute_focus_score(crop_bgr):
    """
    Computes a normalized focus/sharpness score (0 - 100) using Modified Laplacian variance.
    Higher values represent crisp, well-focused stomata. Lower values indicate out-of-focus regions.
    """
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 6 or crop_bgr.shape[1] < 6:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    # Normalize variance to a 0-100 scale
    score = np.clip(variance / 2.0, 0.0, 100.0)
    return round(float(score), 1)


def get_minor_axis_line(polygon):
    """
    Given a shapely Polygon, returns a LineString representing the minor axis 
    that extends well beyond the polygon for cross-sectional intersection.
    """
    rect = cv2.minAreaRect(np.array(polygon.exterior.coords, dtype=np.float32))
    (cx, cy), (w, h), angle = rect

    # In OpenCV minAreaRect, minor axis is the shorter dimension
    if w < h:
        minor_angle = angle
    else:
        minor_angle = angle - 90.0

    angle_rad = np.deg2rad(minor_angle)
    length = max(w, h) * 2.5
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)

    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    return LineString([p1, p2]), angle


def refine_pore_clahe(img_bgr, stoma_poly):
    """
    Stage-2 Crop & Zoom Specialist:
    Crops the stoma with a 20% margin, applies CLAHE, and analyzes the
    medial intensity valley along the minor axis to refine pore width.
    """
    try:
        minx, miny, maxx, maxy = stoma_poly.bounds
        h_box = maxy - miny
        w_box = maxx - minx
        pad = int(max(h_box, w_box) * 0.2)

        H, W = img_bgr.shape[:2]
        y1 = max(0, int(miny) - pad)
        y2 = min(H, int(maxy) + pad)
        x1 = max(0, int(minx) - pad)
        x2 = min(W, int(maxx) + pad)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        # Convert crop to grayscale and apply CLAHE
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Stoma center in crop coordinates
        cx_crop = stoma_poly.centroid.x - x1
        cy_crop = stoma_poly.centroid.y - y1

        # Sample intensity profile along the minor axis
        minor_line, _ = get_minor_axis_line(stoma_poly)
        line_pts = []
        for d in np.linspace(-w_box * 0.4, w_box * 0.4, 60):
            rect = cv2.minAreaRect(np.array(stoma_poly.exterior.coords, dtype=np.float32))
            angle = rect[2] if rect[1][0] < rect[1][1] else rect[2] - 90.0
            rad = np.deg2rad(angle)
            px = int(cx_crop + d * np.cos(rad))
            py = int(cy_crop + d * np.sin(rad))
            if 0 <= px < enhanced.shape[1] and 0 <= py < enhanced.shape[0]:
                line_pts.append((d, enhanced[py, px]))

        if len(line_pts) < 10:
            return None

        distances, intensities = zip(*line_pts)
        intensities = np.array(intensities, dtype=np.float32)

        # Aperture is the dark valley in the center
        min_idx = np.argmin(intensities)
        min_val = intensities[min_idx]
        max_val = np.percentile(intensities, 85)
        valley_depth = max_val - min_val

        if valley_depth > 15:
            # Measure Full Width at Half Maximum (FWHM) of the central valley
            half_thresh = min_val + valley_depth * 0.5
            below_half = np.where(intensities < half_thresh)[0]
            if len(below_half) > 1:
                pore_width_px = abs(distances[below_half[-1]] - distances[below_half[0]])
                return pore_width_px
    except Exception:
        pass
    return None


def predict_tiled(model, img, tile_size=640, stride=448, conf_stoma=0.18, conf_center=0.38, iou_thresh=0.45, margin=16, enhance_contrast=True):
    """
    Runs sliding-window inference with dual-thresholding, tile-edge rejection, and global NMS.
    Returns lists of (stoma_poly, conf) and (center_poly, conf) in full image coordinates.
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        raise ValueError("img must be PIL Image or numpy array")

    # Optional CLAHE contrast enhancement for out-of-focus clarity
    if enhance_contrast:
        img_np = apply_clahe_enhancement(img_np)

    H, W = img_np.shape[:2]
    eval_conf = min(conf_stoma, conf_center)

    # If image fits in single tile, run directly
    if H <= tile_size and W <= tile_size:
        res = model.predict(img_np, conf=eval_conf, verbose=False)[0]
        stomas, centers = [], []
        if res.masks is not None:
            masks = res.masks.xy
            clss = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for m, c, sc in zip(masks, clss, confs):
                if len(m) < 3:
                    continue
                p = Polygon(m)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.geom_type == 'MultiPolygon':
                    p = max(p.geoms, key=lambda g: g.area)
                if p.is_empty or p.geom_type != 'Polygon':
                    continue
                if int(c) == 1 and sc >= conf_stoma:
                    stomas.append((p, float(sc)))
                elif int(c) == 0 and sc >= conf_center:
                    centers.append((p, float(sc)))
        return stomas, centers

    # Generate sliding window tile coordinates
    x_steps = list(range(0, max(1, W - tile_size), stride))
    if not x_steps or x_steps[-1] != W - tile_size:
        x_steps.append(max(0, W - tile_size))

    y_steps = list(range(0, max(1, H - tile_size), stride))
    if not y_steps or y_steps[-1] != H - tile_size:
        y_steps.append(max(0, H - tile_size))

    tiles = []
    tile_coords = []
    for y in y_steps:
        for x in x_steps:
            tile = img_np[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            tile_coords.append((x, y))

    # Batch inference over tiles
    raw_stomas = []
    raw_centers = []
    batch_size = 16

    for i in range(0, len(tiles), batch_size):
        b_tiles = tiles[i:i+batch_size]
        b_coords = tile_coords[i:i+batch_size]
        results = model.predict(b_tiles, conf=eval_conf, verbose=False)

        for res, (tx, ty) in zip(results, b_coords):
            if res.masks is None:
                continue

            masks = res.masks.xy
            clss = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for m, c, sc in zip(masks, clss, confs):
                if len(m) < 3:
                    continue

                min_x = np.min(m[:, 0])
                max_x = np.max(m[:, 0])
                min_y = np.min(m[:, 1])
                max_y = np.max(m[:, 1])

                # Reject detections truncated by internal tile edges (unless at true image borders)
                if tx > 0 and min_x < margin:
                    continue
                if (tx + tile_size < W) and max_x > (tile_size - margin):
                    continue
                if ty > 0 and min_y < margin:
                    continue
                if (ty + tile_size < H) and max_y > (tile_size - margin):
                    continue

                poly_coords = [(pt[0] + tx, pt[1] + ty) for pt in m]
                p = Polygon(poly_coords)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.geom_type == 'MultiPolygon':
                    p = max(p.geoms, key=lambda g: g.area)
                if p.is_empty or p.geom_type != 'Polygon':
                    continue

                # Decoupled confidence thresholds:
                if int(c) == 1 and sc >= conf_stoma:
                    raw_stomas.append((p, float(sc)))
                elif int(c) == 0 and sc >= conf_center:
                    raw_centers.append((p, float(sc)))

    # Global NMS for stomas and centers
    final_stomas = nms_polygons(raw_stomas, iou_thresh=iou_thresh)
    final_centers = nms_polygons(raw_centers, iou_thresh=iou_thresh)

    return final_stomas, final_centers


def nms_polygons(poly_conf_list, iou_thresh=0.45):
    """
    Applies Non-Maximum Suppression to Shapely polygons based on intersection over union.
    """
    if not poly_conf_list:
        return []

    # Sort descending by confidence
    poly_conf_list.sort(key=lambda x: x[1], reverse=True)
    kept = []

    for cand_poly, cand_conf in poly_conf_list:
        cand_area = cand_poly.area
        cand_bounds = cand_poly.bounds
        suppressed = False

        for kept_poly, _ in kept:
            kb = kept_poly.bounds
            if cand_bounds[0] > kb[2] or cand_bounds[2] < kb[0] or cand_bounds[1] > kb[3] or cand_bounds[3] < kb[1]:
                continue

            try:
                inter_area = cand_poly.intersection(kept_poly).area
                if inter_area > 0:
                    union_area = cand_area + kept_poly.area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > iou_thresh:
                        suppressed = True
                        break
            except Exception:
                pass

        if not suppressed:
            kept.append((cand_poly, cand_conf))

    return kept


def process_image(model, img_path_or_array, image_name, tile_size=640, stride=448, conf_stoma=0.18, conf_center=0.38, enhance_contrast=True, min_focus_for_sizing=2.8):
    """
    Main extraction pipeline:
    1. Runs sliding-window tiled detection with dual-thresholding and global NMS.
    2. Enforces strict spatial containment prior (center inside stoma).
    3. Calculates focus quality index for each stoma.
    4. Identifies edge-truncated stomata vs interior stomata.
    5. Categorizes suitability for Stomatal Density (inclusive) vs Ploidy Sizing (strict).
    """
    if isinstance(img_path_or_array, str):
        img_bgr = cv2.imread(img_path_or_array)
        if img_bgr is None:
            return []
    elif isinstance(img_path_or_array, Image.Image):
        img_rgb = np.array(img_path_or_array.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    elif isinstance(img_path_or_array, np.ndarray):
        if len(img_path_or_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_path_or_array, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_path_or_array
    else:
        return []

    H, W = img_bgr.shape[:2]

    # 1. Run tiled inference with decoupled thresholds & optional CLAHE enhancement
    stomas, centers = predict_tiled(
        model, img_bgr, tile_size=tile_size, stride=stride,
        conf_stoma=conf_stoma, conf_center=conf_center,
        enhance_contrast=enhance_contrast
    )

    image_results = []

    for s_idx, (stoma_poly, stoma_conf) in enumerate(stomas):
        rect_stoma = cv2.minAreaRect(np.array(stoma_poly.exterior.coords, dtype=np.float32))
        (cx, cy), (w_s, h_s), angle_s = rect_stoma
        major_axis_s_px = max(w_s, h_s)
        minor_axis_s_px = min(w_s, h_s)

        major_axis_s_um = major_axis_s_px * PX_TO_UM
        minor_axis_s_um = minor_axis_s_px * PX_TO_UM
        stoma_area_um2 = stoma_poly.area * (PX_TO_UM ** 2)
        aspect_ratio_s = major_axis_s_px / max(1e-5, minor_axis_s_px)

        # Check if stoma is truncated by true image perimeter
        edge_margin = 12.0
        minx, miny, maxx, maxy = stoma_poly.bounds
        is_edge_stoma = (minx <= edge_margin or miny <= edge_margin or maxx >= (W - edge_margin) or maxy >= (H - edge_margin))

        # Compute Focus Quality Score (Laplacian variance on stoma crop)
        pad = 6
        cy1 = max(0, int(miny) - pad)
        cy2 = min(H, int(maxy) + pad)
        cx1 = max(0, int(minx) - pad)
        cx2 = min(W, int(maxx) + pad)
        stoma_crop = img_bgr[cy1:cy2, cx1:cx2]
        focus_score = compute_focus_score(stoma_crop)
        is_well_focused = (focus_score >= min_focus_for_sizing)

        # 2. Strict spatial containment: center must be inside the stoma polygon
        matched_center = None
        center_conf = 0.0

        for center_poly, c_conf in centers:
            if stoma_poly.contains(center_poly.centroid) or (stoma_poly.intersection(center_poly).area / max(1e-5, center_poly.area) > 0.65):
                matched_center = center_poly
                center_conf = c_conf
                break

        stage2_refined = False
        pore_major_um = 0.0
        pore_minor_um = 0.0
        pore_area_um2 = 0.0
        guard_cell_width_um = 0.0

        minor_axis_line, _ = get_minor_axis_line(stoma_poly)
        stoma_intersection = minor_axis_line.intersection(stoma_poly)
        stoma_minor_len_px = stoma_intersection.length if not stoma_intersection.is_empty else minor_axis_s_px

        if matched_center is not None:
            # Direct model detection for pore
            rect_center = cv2.minAreaRect(np.array(matched_center.exterior.coords, dtype=np.float32))
            major_axis_c_px = max(rect_center[1][0], rect_center[1][1])
            minor_axis_c_px = min(rect_center[1][0], rect_center[1][1])

            center_intersection = minor_axis_line.intersection(matched_center)
            center_minor_len_px = center_intersection.length if not center_intersection.is_empty else minor_axis_c_px

            pore_major_um = major_axis_c_px * PX_TO_UM
            pore_minor_um = center_minor_len_px * PX_TO_UM
            pore_area_um2 = matched_center.area * (PX_TO_UM ** 2)
            guard_cell_width_um = max(0.0, (stoma_minor_len_px - center_minor_len_px) * PX_TO_UM / 2.0)
        else:
            # Stage-2 CLAHE aperture refinement for closed or faint pore slits
            refined_pore_px = refine_pore_clahe(img_bgr, stoma_poly)
            if refined_pore_px is not None and refined_pore_px < stoma_minor_len_px * 0.6:
                stage2_refined = True
                pore_minor_um = refined_pore_px * PX_TO_UM
                pore_major_um = major_axis_s_um * 0.55
                pore_area_um2 = np.pi * (pore_major_um / 2.0) * (pore_minor_um / 2.0)
                guard_cell_width_um = max(0.0, (stoma_minor_len_px - refined_pore_px) * PX_TO_UM / 2.0)
            else:
                pore_minor_um = 0.0
                pore_major_um = major_axis_s_um * 0.5
                guard_cell_width_um = (stoma_minor_len_px * PX_TO_UM) / 2.0

        aperture_ratio = pore_minor_um / max(1e-5, minor_axis_s_um)

        # General Biological Plausibility (real stoma on epidermis)
        is_plausible = (
            (8.0 <= major_axis_s_um <= 55.0) and
            (1.0 <= aspect_ratio_s <= 3.5) and
            (guard_cell_width_um >= 1.0) and
            (pore_minor_um <= minor_axis_s_um * 0.85)
        )

        # Strict Quality Gate for Ploidy Sizing:
        # Non-truncated, sufficiently sharp, biometrically valid, with a defined pore
        is_suitable_for_ploidy = (
            is_plausible and
            (not is_edge_stoma) and
            is_well_focused and
            (pore_minor_um > 0.0)
        )

        image_results.append({
            "image_name": image_name,
            "stoma_idx": s_idx,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "stoma_conf": round(float(stoma_conf), 3),
            "center_conf": round(float(center_conf), 3),
            "focus_score": focus_score,
            "is_edge_stoma": is_edge_stoma,
            "is_well_focused": is_well_focused,
            "counts_for_density": is_plausible,
            "is_suitable_for_ploidy_sizing": is_suitable_for_ploidy,
            "stoma_major_um": round(float(major_axis_s_um), 2),
            "stoma_minor_um": round(float(minor_axis_s_um), 2),
            "stoma_area_um2": round(float(stoma_area_um2), 2),
            "stoma_aspect_ratio": round(float(aspect_ratio_s), 2),
            "pore_major_um": round(float(pore_major_um), 2),
            "pore_minor_um": round(float(pore_minor_um), 2),
            "pore_area_um2": round(float(pore_area_um2), 2),
            "guard_cell_width_um": round(float(guard_cell_width_um), 2),
            "aperture_ratio": round(float(aperture_ratio), 3),
            "angle_deg": round(float(angle_s), 1),
            "stage2_refined": stage2_refined,
            "is_biometrically_valid": is_plausible,
        })

    return image_results


def extract_traits(model_path, input_dir, output_csv, tile_size=640, stride=448, conf_stoma=0.18, conf_center=0.38, enhance_contrast=True, min_focus_for_sizing=2.8):
    print(f"[*] Loading model from {model_path}...")
    model = YOLO(model_path)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True) +
                         glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True) +
                         glob.glob(os.path.join(input_dir, "**", "*.tif"), recursive=True) +
                         glob.glob(os.path.join(input_dir, "**", "*.tiff"), recursive=True))

    print(f"[*] Found {len(image_paths)} images to process.")
    results_list = []

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(image_paths)}] Processing {img_name}...", flush=True)
        res = process_image(
            model, img_path, img_name,
            tile_size=tile_size, stride=stride,
            conf_stoma=conf_stoma, conf_center=conf_center,
            enhance_contrast=enhance_contrast,
            min_focus_for_sizing=min_focus_for_sizing
        )
        results_list.extend(res)

    if results_list:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False)
        print(f"[+] Successfully extracted {len(df)} stomata measurements -> {output_csv}", flush=True)

        # Generate Image-Level Summary Table for Density & Ploidy Sizing
        summary_rows = []
        for img_name, group in df.groupby("image_name"):
            total_density_count = int(group["counts_for_density"].sum())
            edge_stomata_count = int(group["is_edge_stoma"].sum())
            blurred_stomata_count = int((~group["is_well_focused"]).sum())

            # Standard microscopy field area for 4140x3096 at 0.110767 um/px = ~0.1573 mm2
            field_area_mm2 = 0.1573
            density_per_mm2 = round(total_density_count / field_area_mm2, 1)

            # High-precision subset for ploidy size estimation
            ploidy_subset = group[group["is_suitable_for_ploidy_sizing"]]
            n_ploidy = len(ploidy_subset)

            median_len = round(ploidy_subset["stoma_major_um"].median(), 2) if n_ploidy > 0 else np.nan
            mean_len = round(ploidy_subset["stoma_major_um"].mean(), 2) if n_ploidy > 0 else np.nan
            std_len = round(ploidy_subset["stoma_major_um"].std(), 2) if n_ploidy > 0 else np.nan
            median_width = round(ploidy_subset["stoma_minor_um"].median(), 2) if n_ploidy > 0 else np.nan
            median_gc_width = round(ploidy_subset["guard_cell_width_um"].median(), 2) if n_ploidy > 0 else np.nan
            median_aperture = round(ploidy_subset["pore_minor_um"].median(), 2) if n_ploidy > 0 else np.nan

            summary_rows.append({
                "image_name": img_name,
                "total_stomata_count": total_density_count,
                "stomatal_density_per_mm2": density_per_mm2,
                "edge_stomata_count": edge_stomata_count,
                "blurred_stomata_count": blurred_stomata_count,
                "ploidy_sample_count": n_ploidy,
                "ploidy_median_stoma_length_um": median_len,
                "ploidy_mean_stoma_length_um": mean_len,
                "ploidy_stoma_length_std_um": std_len,
                "ploidy_median_stoma_width_um": median_width,
                "ploidy_median_guard_cell_width_um": median_gc_width,
                "ploidy_median_aperture_um": median_aperture,
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = Path(output_csv).with_name(Path(output_csv).stem + "_image_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"[+] Successfully generated Image-Level Density & Ploidy summary -> {summary_csv}", flush=True)
    else:
        print("[-] No stomata detected.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract stomata geometric traits with tiled inference.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO segmentation model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--tile_size", type=int, default=640, help="Tile size for sliding window")
    parser.add_argument("--stride", type=int, default=448, help="Stride for sliding window")
    parser.add_argument("--conf_stoma", type=float, default=0.18, help="Confidence threshold for stoma detection")
    parser.add_argument("--conf_center", type=float, default=0.38, help="Confidence threshold for pore detection")
    parser.add_argument("--min_focus_for_sizing", type=float, default=2.8, help="Minimum focus score for ploidy sizing (calibrated ~2.8)")
    parser.add_argument("--no_enhance", action="store_true", help="Disable CLAHE contrast enhancement")

    args = parser.parse_args()
    extract_traits(
        args.model, args.input_dir, args.output_csv,
        args.tile_size, args.stride,
        args.conf_stoma, args.conf_center,
        enhance_contrast=(not args.no_enhance),
        min_focus_for_sizing=args.min_focus_for_sizing
    )
