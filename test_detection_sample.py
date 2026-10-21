#!/usr/bin/env python3
"""
test_detection_sample.py
Runs detection & geometric trait measurement on 10 random images from the target folder
using the newly retrained YOLOv8 segmentation model.
"""

import os
import random
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = Path("/home/meow/cesnet_new/Ostatni/Stomata/detect/stomata_trait_comparison_AA012-20251216T171633Z-3-002/stomata_trait_comparison_AA012")
MODEL_PATH = Path("/tmp/stomata_best_latest.pt")
OUTPUT_DIR = Path("/home/meow/Documents/Antigravity/Stomata/test_detection_results")
RANDOM_SEED = 42
N_SAMPLES = 10
CONF_THRESH = 0.25
IMGSZ = 1280
PX_TO_UM = 50.0 / 451.3958

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNO_DIR = OUTPUT_DIR / "annotated_images"
CROPS_DIR = OUTPUT_DIR / "stomata_crops"
ANNO_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


def get_minor_axis_line(polygon):
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda g: g.area)
    rect = cv2.minAreaRect(np.array(polygon.exterior.coords, dtype=np.float32))
    (cx, cy), (w, h), angle = rect

    if w < h:
        minor_angle = angle
    else:
        minor_angle = angle - 90.0

    angle_rad = np.deg2rad(minor_angle)
    length = max(w, h) * 2
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)

    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    return LineString([p1, p2]), (cx, cy), (w, h), angle


def main():
    print(f"[*] Loading retrained model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    # Find all images
    all_images = sorted([
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    print(f"[*] Found {len(all_images)} total images in source directory.")

    random.seed(RANDOM_SEED)
    sample_images = sorted(random.sample(all_images, min(N_SAMPLES, len(all_images))))

    print(f"[*] Selected {len(sample_images)} random images for evaluation:")
    for idx, img_path in enumerate(sample_images, 1):
        print(f"    {idx}. {img_path.name}")

    all_traits = []
    summary_per_image = []

    print("\n" + "=" * 70)
    print(f"{'Image':<25} | {'Detections':<12} | {'Matched Pairs':<15} | {'Avg Length (µm)':<15}")
    print("-" * 70)

    for img_path in sample_images:
        t0 = time.time()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[-] Could not read {img_path.name}")
            continue

        h_img, w_img = img.shape[:2]

        # Run inference
        results = model(
            img,
            conf=CONF_THRESH,
            imgsz=IMGSZ,
            retina_masks=True,
            verbose=False,
        )[0]

        stomas = []
        centers = []

        if results.masks is not None:
            masks = results.masks.xy
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for mask, cls, conf in zip(masks, classes, confs):
                if len(mask) < 3:
                    continue
                poly = Polygon(mask)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue

                if int(cls) == 1:  # stoma
                    stomas.append({"poly": poly, "conf": float(conf), "raw_mask": mask})
                elif int(cls) == 0:  # center / pore
                    centers.append({"poly": poly, "conf": float(conf), "raw_mask": mask})

        # Match stoma to center
        matched_pairs = []
        for s_idx, stoma_data in enumerate(stomas):
            stoma = stoma_data["poly"]
            matched_center = None

            # 1. Point in polygon containment
            for c_data in centers:
                if stoma.contains(c_data["poly"].centroid):
                    matched_center = c_data
                    break

            # 2. Nearest neighbor fallback if within reasonable distance
            if not matched_center and centers:
                dists = [stoma.centroid.distance(c["poly"].centroid) for c in centers]
                min_dist_idx = np.argmin(dists)
                if dists[min_dist_idx] < 120:  # pixels
                    matched_center = centers[min_dist_idx]

            if matched_center:
                matched_pairs.append((stoma_data, matched_center))

        # Create overlay visualization
        overlay = img.copy()
        img_vis = img.copy()

        image_measurements = []

        for pair_idx, (s_data, c_data) in enumerate(matched_pairs):
            stoma_poly = s_data["poly"]
            center_poly = c_data["poly"]

            line_geom, (cx, cy), (w_box, h_box), angle = get_minor_axis_line(stoma_poly)

            stoma_intersect = line_geom.intersection(stoma_poly)
            center_intersect = line_geom.intersection(center_poly)

            stoma_minor_px = stoma_intersect.length
            center_minor_px = center_intersect.length
            guard_cell_px = (stoma_minor_px - center_minor_px) / 2.0

            rect = cv2.minAreaRect(np.array(stoma_poly.exterior.coords, dtype=np.float32))
            stoma_major_px = max(rect[1][0], rect[1][1])

            trait_entry = {
                "image_name": img_path.name,
                "stoma_id": pair_idx + 1,
                "stoma_conf": round(s_data["conf"], 3),
                "pore_conf": round(c_data["conf"], 3),
                "cx": int(cx),
                "cy": int(cy),
                "major_axis_px": round(stoma_major_px, 1),
                "major_axis_um": round(stoma_major_px * PX_TO_UM, 2),
                "minor_axis_px": round(stoma_minor_px, 1),
                "minor_axis_um": round(stoma_minor_px * PX_TO_UM, 2),
                "pore_width_px": round(center_minor_px, 1),
                "pore_width_um": round(center_minor_px * PX_TO_UM, 2),
                "guard_cell_width_px": round(guard_cell_px, 1),
                "guard_cell_width_um": round(guard_cell_px * PX_TO_UM, 2),
            }
            all_traits.append(trait_entry)
            image_measurements.append(trait_entry)

            # Draw visual layers
            s_pts = np.array(stoma_poly.exterior.coords, dtype=np.int32)
            c_pts = np.array(center_poly.exterior.coords, dtype=np.int32)

            # 1. Stoma outline (Green)
            cv2.polylines(img_vis, [s_pts], True, (0, 255, 0), 3)

            # 2. Pore glow on overlay (Orange)
            cv2.fillPoly(overlay, [c_pts], (0, 140, 255))
            cv2.polylines(img_vis, [c_pts], True, (0, 140, 255), 2)

            # 3. Minor axis line (Magenta)
            if not stoma_intersect.is_empty:
                if stoma_intersect.geom_type == "LineString":
                    coords = list(stoma_intersect.coords)
                    p_start = (int(coords[0][0]), int(coords[0][1]))
                    p_end = (int(coords[-1][0]), int(coords[-1][1]))
                    cv2.line(img_vis, p_start, p_end, (255, 0, 255), 2)

            # 4. Pore width line (Yellow)
            if not center_intersect.is_empty:
                if center_intersect.geom_type == "LineString":
                    coords = list(center_intersect.coords)
                    p_start = (int(coords[0][0]), int(coords[0][1]))
                    p_end = (int(coords[-1][0]), int(coords[-1][1]))
                    cv2.line(img_vis, p_start, p_end, (0, 255, 255), 3)

            # 5. ID label
            cv2.putText(
                img_vis,
                f"#{pair_idx+1} ({trait_entry['major_axis_um']}um)",
                (int(cx) - 40, max(20, int(cy) - 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Save detailed zoom crop for close inspection (up to 3 crops per image)
            if pair_idx < 3:
                crop_half = 180
                crop_x1 = max(0, int(cx) - crop_half)
                crop_y1 = max(0, int(cy) - crop_half)
                crop_x2 = min(w_img, int(cx) + crop_half)
                crop_y2 = min(h_img, int(cy) + crop_half)

                crop_img = img_vis[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_name = f"{img_path.stem}_stoma_{pair_idx+1}.jpg"
                cv2.imwrite(str(CROPS_DIR / crop_name), crop_img)

        # Merge overlay transparency
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img_vis, 1 - alpha, 0, img_vis)

        # Save annotated image
        out_anno_path = ANNO_DIR / f"{img_path.stem}_annotated.jpg"
        # Resize full field for reasonable disk viewing if huge, while keeping full quality
        cv2.imwrite(str(out_anno_path), img_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        n_stomas = len(stomas)
        n_centers = len(centers)
        n_matched = len(matched_pairs)
        avg_len = (
            np.mean([m["major_axis_um"] for m in image_measurements])
            if image_measurements
            else 0.0
        )

        summary_per_image.append({
            "image": img_path.name,
            "stomas_detected": n_stomas,
            "pores_detected": n_centers,
            "matched_pairs": n_matched,
            "avg_length_um": round(avg_len, 2),
            "inference_time_s": round(time.time() - t0, 2),
        })

        print(
            f"{img_path.name:<25} | {f'{n_stomas}s / {n_centers}p':<12} | "
            f"{n_matched:<15} | {avg_len:<15.2f}"
        )

    print("-" * 70)

    df_traits = pd.DataFrame(all_traits)
    csv_traits_path = OUTPUT_DIR / "stomata_measurements_sample10.csv"
    df_traits.to_csv(csv_traits_path, index=False)

    df_summary = pd.DataFrame(summary_per_image)
    csv_summary_path = OUTPUT_DIR / "image_summary_sample10.csv"
    df_summary.to_csv(csv_summary_path, index=False)

    print(f"\n[+] Results saved:")
    print(f"    Traits CSV:      {csv_traits_path}")
    print(f"    Summary CSV:     {csv_summary_path}")
    print(f"    Annotated Imgs:  {ANNO_DIR}")
    print(f"    Stomata Crops:   {CROPS_DIR}")

    print("\n" + "=" * 60)
    print("STATISTICAL TRAIT SUMMARY ACROSS ALL DETECTIONS")
    print("=" * 60)
    if not df_traits.empty:
        stats = df_traits[[
            "major_axis_um",
            "minor_axis_um",
            "pore_width_um",
            "guard_cell_width_um",
            "stoma_conf",
            "pore_conf",
        ]].describe()
        print(stats.round(2))
    else:
        print("No valid stomata detected.")


if __name__ == "__main__":
    main()
