#!/usr/bin/env python3
"""
generate_roboflow_preannotations.py
Generates 200 curated, model-assisted pre-annotated 640x640 crops for Roboflow.
Prioritizes challenging crops focused strictly on:
- Low-confidence detections (stoma or pore confidence between 0.15 and 0.45)
- Stomata with closed or faint pores (providing whole stoma outlines + medial slit proposals)
- Focal blur gradients and dense overlapping clusters
- Negative background distractor tiles (15 crops) from /home/meow/cesnet_new/Ostatni/Stomata/backgrounds
Uses targeted coordinate mining from detection dataset across top challenging micrographs for rapid generation.
"""

import os
import glob
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from ultralytics import YOLO

# ── Paths & Config ─────────────────────────────────────────────────────────────
WORKSPACE_DIR = Path("/home/meow/Documents/Antigravity/Stomata")
MODEL_PATH = Path("/home/meow/cesnet_new/Ostatni/Stomata/trained_models/stomata_best_latest.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("/tmp/stomata_best_latest.pt")

STOMATA_CSV = WORKSPACE_DIR / "data/stomata_traits_all_3253_stomata.csv"
SOURCE_BASE = Path("/home/meow/cesnet_new/Ostatni/Stomata/extracted/stomata_trait_comparison")
OUTPUT_DIR = WORKSPACE_DIR / "roboflow_preannotated_dataset"

TOTAL_TARGET = 200
N_BACKGROUND_TARGET = 15
CROP_SIZE = 640
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def polygon_to_yolo_str(poly, class_id, crop_w=640, crop_h=640):
    """Convert shapely polygon to YOLO normalized segmentation string."""
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda g: g.area)
    coords = list(poly.exterior.coords)
    if len(coords) < 3:
        return None

    simplified = poly.simplify(1.0, preserve_topology=True)
    if simplified.geom_type == 'Polygon' and len(simplified.exterior.coords) >= 3:
        coords = list(simplified.exterior.coords)

    norm_coords = []
    for x, y in coords[:-1]:
        nx = max(0.0, min(1.0, x / crop_w))
        ny = max(0.0, min(1.0, y / crop_h))
        norm_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

    if len(norm_coords) < 6:
        return None

    return f"{class_id} " + " ".join(norm_coords)


def main():
    print("=" * 70, flush=True)
    print("  Roboflow Model-Assisted Pre-Annotation Generator (Targeted Low-Conf Mining)", flush=True)
    print("=" * 70, flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    print(f"[*] Loading retrained model from {MODEL_PATH}...", flush=True)
    model = YOLO(str(MODEL_PATH))

    # 1. Index all image files
    print("[*] Indexing micrograph file paths on CESNET mount...", flush=True)
    img_map = {p.name: p for p in SOURCE_BASE.rglob("*.*") if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')}
    print(f"[+] Found {len(img_map)} total micrograph files.", flush=True)

    # 2. Query hardest stomata from detection dataset
    print(f"[*] Reading full detection dataset from {STOMATA_CSV}...", flush=True)
    df_stoma = pd.read_csv(STOMATA_CSV, low_memory=False)

    df_stoma['is_hard'] = (df_stoma['stoma_conf'] < 0.50) | (df_stoma['center_conf'] < 0.45) | (df_stoma['pore_minor_um'] == 0)

    # Group by image to find images with the highest density of challenging stomata
    hard_per_img = df_stoma[df_stoma['is_hard']].groupby('image_name').size().sort_values(ascending=False)
    top_images = hard_per_img.head(45).index.tolist()

    df_top = df_stoma[df_stoma['image_name'].isin(top_images) & df_stoma['is_hard']].copy()
    df_top['priority'] = (
        (df_top['stoma_conf'] < 0.45).astype(int) * 3.5 +
        ((df_top['center_conf'] > 0) & (df_top['center_conf'] < 0.45)).astype(int) * 2.5 +
        (df_top['pore_minor_um'] == 0).astype(int) * 2.0 +
        ((df_top['focus_score'] > 0) & (df_top['focus_score'] < 3.5)).astype(int) * 1.5 +
        (1.0 - df_top['stoma_conf']) * 2.0
    )
    df_top = df_top.sort_values('priority', ascending=False)

    # Spatially separated crop selection (max 6 crops per image, distance >= 300px)
    n_hard_target = TOTAL_TARGET - N_BACKGROUND_TARGET
    selected_targets = []
    per_img_pts = {}

    for _, row in df_top.iterrows():
        if len(selected_targets) >= n_hard_target:
            break
        img = row['image_name']
        cx, cy = float(row['centroid_x']), float(row['centroid_y'])
        pts = per_img_pts.get(img, [])
        if len(pts) >= 6:
            continue
        if not any(np.hypot(cx - px, cy - py) < 300 for px, py in pts):
            pts.append((cx, cy))
            per_img_pts[img] = pts
            selected_targets.append(row)

    print(f"[+] Selected {len(selected_targets)} spatially separated low-confidence crops across {len(per_img_pts)} prioritized micrographs.", flush=True)

    # 3. Prepare output directories
    img_out = OUTPUT_DIR / "images"
    lbl_out = OUTPUT_DIR / "labels"
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    log_records = []
    crop_counter = 0

    # Group selected targets by image to read each micrograph only once
    by_image = {}
    for row in selected_targets:
        img_name = row['image_name']
        by_image.setdefault(img_name, []).append(row)

    print(f"[*] Processing {len(by_image)} micrographs and generating pre-annotations...", flush=True)
    for img_idx, (img_name, rows) in enumerate(by_image.items(), 1):
        img_path = img_map.get(img_name)
        if img_path is None or not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]

        for row in rows:
            cx = float(row['centroid_x'])
            cy = float(row['centroid_y'])

            # Clamp bounding box
            x1 = int(np.clip(cx - CROP_SIZE // 2, 0, max(0, W - CROP_SIZE)))
            y1 = int(np.clip(cy - CROP_SIZE // 2, 0, max(0, H - CROP_SIZE)))
            x2 = x1 + CROP_SIZE
            y2 = y1 + CROP_SIZE

            crop = img_bgr[y1:y2, x1:x2]
            if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
                continue

            results = model.predict(crop, conf=0.15, verbose=False)[0]

            stoma_polys = []
            pore_polys = []

            if results.masks is not None:
                masks = results.masks.xy
                classes = results.boxes.cls.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                for m, c, conf in zip(masks, classes, confs):
                    if len(m) < 3:
                        continue
                    p = Polygon(m)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.geom_type == 'MultiPolygon':
                        p = max(p.geoms, key=lambda g: g.area)
                    if p.is_empty or p.geom_type != 'Polygon':
                        continue

                    if int(c) == 1:
                        stoma_polys.append((p, float(conf)))
                    elif int(c) == 0:
                        pore_polys.append((p, float(conf)))

            lines = []
            matched_pores = set()

            for s_poly, s_conf in stoma_polys:
                # Class 1: whole stoma
                s_str = polygon_to_yolo_str(s_poly, class_id=1, crop_w=CROP_SIZE, crop_h=CROP_SIZE)
                if s_str:
                    lines.append(s_str)

                # Class 0: pore
                found_pore = False
                for p_idx, (p_poly, p_conf) in enumerate(pore_polys):
                    if p_idx in matched_pores:
                        continue
                    if s_poly.contains(p_poly.centroid) or (s_poly.intersection(p_poly).area / max(1.0, p_poly.area) > 0.5):
                        p_str = polygon_to_yolo_str(p_poly, class_id=0, crop_w=CROP_SIZE, crop_h=CROP_SIZE)
                        if p_str:
                            lines.append(p_str)
                            matched_pores.add(p_idx)
                            found_pore = True
                            break

                # If pore is closed / missing, provide medial longitudinal groove guide
                if not found_pore:
                    coords = np.array(s_poly.exterior.coords, dtype=np.float32)
                    (box_cx, box_cy), (w, h), angle = cv2.minAreaRect(coords)
                    pore_w = max(3.0, min(w, h) * 0.16)
                    pore_l = max(w, h) * 0.45
                    p_box = cv2.boxPoints(((box_cx, box_cy), (min(pore_w, pore_l), max(pore_w, pore_l)), angle))
                    p_poly_est = Polygon(p_box)
                    if s_poly.contains(p_poly_est.centroid):
                        p_str = polygon_to_yolo_str(p_poly_est, class_id=0, crop_w=CROP_SIZE, crop_h=CROP_SIZE)
                        if p_str:
                            lines.append(p_str)

            crop_counter += 1
            fname_stem = f"stomata_crop_{crop_counter:03d}_{row['cytotype']}_{row['plant_ID']}_{row['leaf_side']}_x{x1}_y{y1}"

            cv2.imwrite(str(img_out / f"{fname_stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lbl_out / f"{fname_stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

            confs = [c for _, c in stoma_polys] if stoma_polys else [float(row['stoma_conf'])]
            log_records.append({
                "crop_id": crop_counter,
                "filename": f"{fname_stem}.jpg",
                "type": "targeted_low_conf",
                "cytotype": row['cytotype'],
                "population": row['population'],
                "stomata_count": len(stoma_polys),
                "pores_count": len(matched_pores),
                "min_conf": round(float(min(confs)), 3),
                "avg_stoma_conf": round(float(sum(confs) / len(confs)), 3),
                "target_stoma_conf": round(float(row['stoma_conf']), 3),
                "target_center_conf": round(float(row['center_conf']), 3),
                "target_focus_score": round(float(row.get('focus_score', 0)), 1),
            })

        print(f"  [{img_idx:02d}/{len(by_image)}] {img_name} -> Total crops so far: {crop_counter}", flush=True)

    # 4. Add Background Distractor Crops (Negative Controls)
    print(f"[*] Adding {N_BACKGROUND_TARGET} pure background negative control crops...", flush=True)
    bg_dir = Path("/home/meow/cesnet_new/Ostatni/Stomata/backgrounds")
    if bg_dir.exists():
        bg_files = sorted(list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")))
        random.shuffle(bg_files)
        for bg_file in bg_files[:N_BACKGROUND_TARGET]:
            bg_img = cv2.imread(str(bg_file))
            if bg_img is not None:
                h, w = bg_img.shape[:2]
                if h >= CROP_SIZE and w >= CROP_SIZE:
                    cy, cx = h // 2, w // 2
                    crop = bg_img[cy - CROP_SIZE//2: cy + CROP_SIZE//2, cx - CROP_SIZE//2: cx + CROP_SIZE//2]
                else:
                    crop = cv2.resize(bg_img, (CROP_SIZE, CROP_SIZE))

                crop_counter += 1
                fname_stem = f"stomata_crop_{crop_counter:03d}_bg_control_{crop_counter}"
                cv2.imwrite(str(img_out / f"{fname_stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(lbl_out / f"{fname_stem}.txt", "w") as f:
                    pass

                log_records.append({
                    "crop_id": crop_counter,
                    "filename": f"{fname_stem}.jpg",
                    "type": "background_negative",
                    "cytotype": "background",
                    "population": "negative",
                    "stomata_count": 0,
                    "pores_count": 0,
                    "min_conf": 0.0,
                    "avg_stoma_conf": 0.0,
                    "target_stoma_conf": 0.0,
                    "target_center_conf": 0.0,
                    "target_focus_score": 0.0,
                })

    # 5. Write data.yaml for Roboflow
    data_yaml_content = f"""# Roboflow / YOLOv8 / YOLO11 Segmentation Dataset Configuration
# Generated with Active Learning Targeted on Low-Confidence Detections
train: ./images
val: ./images

nc: 2
names: ['center', 'stomata-huw1']

roboflow:
  project: stomata-retraining
  description: "200 targeted low-confidence, closed-pore, and blurred crops with model-assisted pre-annotations"
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    # 6. Write metadata CSV & README
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(OUTPUT_DIR / "crops_metadata_summary.csv", index=False)

    readme_content = f"""# Roboflow Model-Assisted Pre-Annotated Dataset (Targeted Low-Confidence Mining)

This dataset contains **{len(log_df)} curated 640x640 microscopy crops** selected specifically for active-learning annotation in **Roboflow**.

### Dataset Composition
- **Total Crops:** {len(log_df)}
- **Targeted Low-Confidence Stomata Crops:** {len(log_df[log_df['type'] == 'targeted_low_conf'])} (each crop centered on historically lowest-confidence or closed-pore stomata)
- **Background Distractor Crops:** {len(log_df[log_df['type'] == 'background_negative'])} (cuticle folds, cell wall junctions, and hair scars with zero stomata)
- **Classes:**
  - `0`: `center` (pore aperture slit)
  - `1`: `stomata-huw1` (whole stoma / guard cell pair)

### Quick Roboflow Import Instructions
1. Upload `roboflow_preannotated_dataset.zip` directly to your Roboflow project.
2. Select format: **YOLOv8 Segmentation** or **YOLO11 Segmentation**.
3. All crops are pre-segmented: simply accept or adjust the boundaries in the Roboflow annotation UI.
"""
    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write(readme_content)

    # 7. Create Zip Archive
    zip_path = WORKSPACE_DIR / "roboflow_preannotated_dataset.zip"
    print(f"\n[*] Creating zip archive at {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                fpath = os.path.join(root, file)
                arcname = os.path.relpath(fpath, OUTPUT_DIR)
                zipf.write(fpath, arcname)

    print(f"[+] Zip file created successfully ({os.path.getsize(zip_path):,} bytes)", flush=True)

    cesnet_backup = Path("/home/meow/cesnet_new/Ostatni/Stomata/crops_for_labeling")
    if cesnet_backup.exists():
        cesnet_dest = cesnet_backup / "roboflow_preannotated_200.zip"
        shutil.copy(zip_path, cesnet_dest)
        print(f"[+] Backed up zip to CESNET S3 mount: {cesnet_dest}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"  FINISHED: {len(log_df)} targeted crops generated with YOLO pre-annotations!", flush=True)
    print(f"  Folder: {OUTPUT_DIR}", flush=True)
    print(f"  Zip:    {zip_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
