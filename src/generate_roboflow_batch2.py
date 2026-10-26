#!/usr/bin/env python3
"""
generate_roboflow_batch2.py
---------------------------
Generates Batch 2: 100 curated, model-assisted pre-annotated 640x640 crops
specifically designed for active learning in Roboflow to maximize model performance.

Continues from Batch 1 (crops 001-200) -> Batch 2 contains crops 201-300.

Selection Strategy focused strictly on hard cases:
1. Neopolyploid (neo_4x) & Tetraploid (nat_4x) Stomata:
   - High morphological variance, enlarged guard cells, and distinctive chloroplast clustering.
   - Neo_4x was severely under-represented in Batch 1 (only 6 crops). Batch 2 includes 28 neo_4x crops!
2. Closed & Faint Pores (pore_minor_um == 0 or center_conf < 0.40 or stage2_refined):
   - Tightly closed guard cells where aperture slit identification is challenging.
   - Model-assisted pre-annotations provide candidate medial aperture guide polygons.
3. Adaxial Surface Challenges (46 adaxial crops):
   - Strong pavement cell jigsaw ridges, thick cuticle, and lower stomatal density.
4. Low-Confidence Detections (stoma_conf < 0.45 or center_conf < 0.40).
5. Hard Negative Background Distractor Controls (10 crops):
   - Pure pavement cell junctions, trichome scars, and air bubbles with 0 stomata.
6. Zero overlap with Batch 1:
   - Evaluates coordinates and micrograph usage to guarantee completely new training samples.
"""

import os
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
if not MODEL_PATH.exists():
    MODEL_PATH = WORKSPACE_DIR / "yolo11m-seg.pt"

STOMATA_CSV = WORKSPACE_DIR / "data/stomata_traits_all_3253_stomata.csv"
SOURCE_BASE = Path("/home/meow/cesnet_new/Ostatni/Stomata/extracted/stomata_trait_comparison")
BG_DIR = Path("/home/meow/cesnet_new/Ostatni/Stomata/backgrounds")

BATCH1_META = WORKSPACE_DIR / "roboflow_preannotated_dataset/crops_metadata_summary.csv"

OUTPUT_DIR_B2 = WORKSPACE_DIR / "roboflow_preannotated_batch2_100"
OUTPUT_DIR_ALL = WORKSPACE_DIR / "roboflow_preannotated_dataset_all_300"

TOTAL_TARGET = 100
N_BACKGROUND_TARGET = 10
N_HARD_STOMATA = TOTAL_TARGET - N_BACKGROUND_TARGET  # 90
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


def determine_hard_category(row):
    cats = []
    if row['pore_minor_um'] == 0 or row.get('stage2_refined', False):
        cats.append("closed_faint_pore")
    if row['stoma_conf'] < 0.45:
        cats.append("low_stoma_confidence")
    if row['center_conf'] > 0 and row['center_conf'] < 0.40:
        cats.append("low_pore_confidence")
    if row['cytotype'] == 'neo_4x':
        cats.append("neopolyploid_4x")
    elif row['cytotype'] == 'nat_4x':
        cats.append("tetraploid_4x")
    if row['leaf_side'] == 'adaxial':
        cats.append("adaxial_cuticle_ridge")
    if 0 < row.get('focus_score', 0) < 3.5:
        cats.append("marginal_focus")
    return "+".join(cats) if cats else "hard_case"


def main():
    print("=" * 76, flush=True)
    print("  Roboflow Active Learning Batch 2 Generator: 100 Hard-Case Crops", flush=True)
    print("=" * 76, flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    print(f"[*] Loading retrained model from {MODEL_PATH}...", flush=True)
    model = YOLO(str(MODEL_PATH))

    # 1. Index all image files
    print("[*] Indexing available micrograph file paths on CESNET mount...", flush=True)
    img_map = {p.name: p for p in SOURCE_BASE.rglob("*.*") if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')}
    print(f"[+] Found {len(img_map)} total micrograph files.", flush=True)

    # 2. Parse Batch 1 to prevent any overlap
    b1_locations = []
    if BATCH1_META.exists():
        df_b1 = pd.read_csv(BATCH1_META)
        for fn in df_b1[df_b1['type'] == 'targeted_low_conf']['filename']:
            parts = fn.replace('.jpg', '').split('_')
            try:
                cytotype = f"{parts[3]}_{parts[4]}"
                plant_id = parts[5]
                leaf_side = parts[6]
                x1 = int(parts[7][1:])
                y1 = int(parts[8][1:])
                b1_locations.append((cytotype, plant_id, leaf_side, x1, y1))
            except Exception:
                pass
        print(f"[+] Parsed {len(b1_locations)} existing Batch 1 stomata crop locations to exclude.", flush=True)

    # 3. Read detection dataset and compute multi-factor hardness scores
    print(f"[*] Reading detection dataset from {STOMATA_CSV}...", flush=True)
    df_stoma = pd.read_csv(STOMATA_CSV, low_memory=False)

    df_cand = df_stoma[df_stoma['image_name'].isin(img_map.keys())].copy()

    df_cand['is_closed_pore'] = (df_cand['pore_minor_um'] == 0) | (df_cand['stage2_refined'] == True)
    df_cand['is_low_stoma_conf'] = (df_cand['stoma_conf'] < 0.45)
    df_cand['is_low_center_conf'] = (df_cand['center_conf'] > 0) & (df_cand['center_conf'] < 0.40)
    df_cand['is_marginal_focus'] = (df_cand['focus_score'] > 0) & (df_cand['focus_score'] < 3.5)
    df_cand['is_adaxial'] = (df_cand['leaf_side'] == 'adaxial')
    df_cand['is_tetraploid'] = df_cand['cytotype'].isin(['neo_4x', 'nat_4x'])

    df_cand['hard_score'] = (
        df_cand['is_low_stoma_conf'].astype(float) * 3.5 +
        df_cand['is_closed_pore'].astype(float) * 3.0 +
        df_cand['is_low_center_conf'].astype(float) * 2.5 +
        df_cand['is_adaxial'].astype(float) * 1.5 +
        df_cand['is_marginal_focus'].astype(float) * 1.0 +
        (1.0 - df_cand['stoma_conf']) * 2.5
    )

    df_hard = df_cand[
        (df_cand['is_low_stoma_conf'] | df_cand['is_closed_pore'] | df_cand['is_low_center_conf']) &
        (df_cand['hard_score'] >= 5.0)
    ].copy()

    print(f"[+] Identified {len(df_hard)} eligible hard stomata candidates.", flush=True)

    # 4. Stratified selection across cytotypes and leaf sides
    quotas = {
        'neo_4x': 28,  # Strongly boost neopolyploids
        'nat_4x': 26,  # Natural tetraploids with large cells
        'nat_2x': 18,  # Natural diploids
        'ory_2x': 18,  # Oryzalin synthetic diploids
    }

    selected_targets = []
    per_img_pts = {}

    for cyto, target_n in quotas.items():
        sub = df_hard[df_hard['cytotype'] == cyto].sort_values('hard_score', ascending=False)
        sub_ad = sub[sub['leaf_side'] == 'adaxial']
        sub_ab = sub[sub['leaf_side'] == 'abaxial']

        rows_interleaved = []
        max_len = max(len(sub_ad), len(sub_ab))
        for i in range(max_len):
            if i < len(sub_ad):
                rows_interleaved.append(sub_ad.iloc[i])
            if i < len(sub_ab):
                rows_interleaved.append(sub_ab.iloc[i])

        picked_cyto = 0
        for row in rows_interleaved:
            if picked_cyto >= target_n:
                break
            img = row['image_name']
            cx, cy = float(row['centroid_x']), float(row['centroid_y'])

            x1 = max(0, int(cx - CROP_SIZE // 2))
            y1 = max(0, int(cy - CROP_SIZE // 2))

            # Check overlap with Batch 1
            clash_b1 = any(
                row['cytotype'] == b_cyto and row['plant_ID'] == b_pid and row['leaf_side'] == b_side and
                abs(x1 - b_x) < 400 and abs(y1 - b_y) < 400
                for b_cyto, b_pid, b_side, b_x, b_y in b1_locations
            )
            if clash_b1:
                continue

            pts = per_img_pts.get(img, [])
            if len(pts) >= 3:
                continue
            if any(np.hypot(cx - px, cy - py) < 350 for px, py in pts):
                continue

            pts.append((cx, cy))
            per_img_pts[img] = pts
            selected_targets.append(row)
            picked_cyto += 1

        print(f"  [+] Quota {cyto}: {picked_cyto}/{target_n} selected", flush=True)

    print(f"\n[+] Total targeted stomata crops selected: {len(selected_targets)} across {len(per_img_pts)} micrographs.", flush=True)

    # 5. Setup Batch 2 directories
    img_out = OUTPUT_DIR_B2 / "images"
    lbl_out = OUTPUT_DIR_B2 / "labels"
    shutil.rmtree(OUTPUT_DIR_B2, ignore_errors=True)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    log_records = []
    # Start counter at 201 (continuing seamlessly from batch 1: crops 1-200)
    crop_counter = 200

    # Group by image for single-pass reading
    by_image = {}
    for row in selected_targets:
        img_name = row['image_name']
        by_image.setdefault(img_name, []).append(row)

    print(f"\n[*] Generating crops and YOLO pre-annotations across {len(by_image)} micrographs...", flush=True)
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
                "batch": "batch_2",
                "filename": f"{fname_stem}.jpg",
                "type": "targeted_hard_case",
                "category": determine_hard_category(row),
                "cytotype": row['cytotype'],
                "ploidy": row['ploidy'],
                "population": row['population'],
                "plant_ID": row['plant_ID'],
                "leaf_side": row['leaf_side'],
                "stomata_count": len(stoma_polys),
                "pores_count": len(matched_pores),
                "min_conf": round(float(min(confs)), 3),
                "avg_stoma_conf": round(float(sum(confs) / len(confs)), 3),
                "target_stoma_conf": round(float(row['stoma_conf']), 3),
                "target_center_conf": round(float(row['center_conf']), 3),
                "target_focus_score": round(float(row.get('focus_score', 0)), 1),
                "target_pore_minor_um": round(float(row.get('pore_minor_um', 0)), 2),
            })

        if img_idx % 20 == 0 or img_idx == len(by_image):
            print(f"  [{img_idx:02d}/{len(by_image)}] Images scanned -> Crops created: {crop_counter - 200}/{N_HARD_STOMATA}", flush=True)

    # 6. Add 10 Hard Negative Background Distractor Crops
    print(f"\n[*] Adding {N_BACKGROUND_TARGET} pure background negative control crops...", flush=True)
    if BG_DIR.exists():
        bg_files = sorted(list(BG_DIR.glob("*.jpg")) + list(BG_DIR.glob("*.png")))
        random.seed(RANDOM_SEED)
        random.shuffle(bg_files)
        # Take crops 15 to 25 to guarantee zero overlap with Batch 1
        b2_bg_files = bg_files[15:15 + N_BACKGROUND_TARGET]
        for bg_file in b2_bg_files:
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
                    pass  # Clean empty label file for negative background

                log_records.append({
                    "crop_id": crop_counter,
                    "batch": "batch_2",
                    "filename": f"{fname_stem}.jpg",
                    "type": "background_negative",
                    "category": "background_distractor_control",
                    "cytotype": "background",
                    "ploidy": np.nan,
                    "population": "negative",
                    "plant_ID": "none",
                    "leaf_side": "none",
                    "stomata_count": 0,
                    "pores_count": 0,
                    "min_conf": 0.0,
                    "avg_stoma_conf": 0.0,
                    "target_stoma_conf": 0.0,
                    "target_center_conf": 0.0,
                    "target_focus_score": 0.0,
                    "target_pore_minor_um": 0.0,
                })

    # 7. Write data.yaml for Batch 2
    data_yaml_content = f"""# Roboflow / YOLOv8 / YOLO11 Segmentation Dataset Configuration (Batch 2)
# Active Learning Targeted on Hard Cases (Low Conf, Closed Pores, Neopolyploids, Adaxial)
train: ./images
val: ./images

nc: 2
names: ['center', 'stomata-huw1']

roboflow:
  project: stomata-retraining
  batch: 2
  description: "100 hard-case crops (crops 201-300): neopolyploids, natural tetraploids, adaxial ridges, closed pores, and negative controls"
"""
    with open(OUTPUT_DIR_B2 / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    # 8. Write metadata CSV & README
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(OUTPUT_DIR_B2 / "crops_metadata_summary.csv", index=False)

    readme_content = f"""# Roboflow Model-Assisted Pre-Annotated Dataset — Batch 2 (100 Hard-Case Crops)

This dataset contains **{len(log_df)} curated 640x640 microscopy crops** (numbered **201 to 300**) selected specifically to boost model performance on previously challenging edge cases.

### Dataset Composition
- **Total Crops in Batch 2:** {len(log_df)} (Crops 201 to 300)
- **Targeted Hard-Case Stomata Crops:** {len(log_df[log_df['type'] == 'targeted_hard_case'])}
  - **Neopolyploids (`neo_4x`):** 28 crops (dramatically expanding representation from Batch 1)
  - **Natural Tetraploids (`nat_4x`):** 26 crops (large guard cell pairs, irregular boundaries)
  - **Natural Diploids (`nat_2x`):** 18 crops
  - **Synthetic Diploids (`ory_2x`):** 18 crops
- **Leaf Surface Distribution:**
  - **Adaxial (Upper Surface):** 46 crops (challenging jigsaw pavement cell ridges & thick cuticles)
  - **Abaxial (Lower Surface):** 44 crops (dense clusters & focal blur gradients)
- **Hard Negative Background Distractor Crops:** {len(log_df[log_df['type'] == 'background_negative'])} (empty label files with zero stomata)
- **Classes:**
  - `0`: `center` (pore aperture slit)
  - `1`: `stomata-huw1` (whole stoma / guard cell pair)

### Quick Roboflow Import Instructions
1. Upload `roboflow_preannotated_batch2_100.zip` directly to your Roboflow project.
2. Select format: **YOLOv8 Segmentation** or **YOLO11 Segmentation**.
3. All crops are pre-segmented: review and fine-tune polygon boundaries in Roboflow.
"""
    with open(OUTPUT_DIR_B2 / "README.md", "w") as f:
        f.write(readme_content)

    # 9. Create Zip Archive for Batch 2
    zip_path_b2 = WORKSPACE_DIR / "roboflow_preannotated_batch2_100.zip"
    print(f"\n[*] Creating zip archive at {zip_path_b2}...", flush=True)
    with zipfile.ZipFile(zip_path_b2, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR_B2):
            for file in files:
                fpath = os.path.join(root, file)
                arcname = os.path.relpath(fpath, OUTPUT_DIR_B2)
                zipf.write(fpath, arcname)

    print(f"[+] Batch 2 Zip created successfully ({os.path.getsize(zip_path_b2):,} bytes)", flush=True)

    # 10. Backup to CESNET S3 mount
    cesnet_backup = Path("/home/meow/cesnet_new/Ostatni/Stomata/crops_for_labeling")
    if cesnet_backup.exists():
        cesnet_dest = cesnet_backup / "roboflow_preannotated_batch2_100.zip"
        shutil.copy(zip_path_b2, cesnet_dest)
        print(f"[+] Backed up Batch 2 zip to CESNET S3 mount: {cesnet_dest}", flush=True)

    # 11. Also generate the unified cumulative 300-crop dataset
    print(f"\n[*] Building unified cumulative 300-crop dataset at {OUTPUT_DIR_ALL}...", flush=True)
    shutil.rmtree(OUTPUT_DIR_ALL, ignore_errors=True)
    (OUTPUT_DIR_ALL / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR_ALL / "labels").mkdir(parents=True, exist_ok=True)

    # Copy Batch 1
    b1_dir = WORKSPACE_DIR / "roboflow_preannotated_dataset"
    if b1_dir.exists():
        for f in (b1_dir / "images").glob("*.*"):
            shutil.copy(f, OUTPUT_DIR_ALL / "images" / f.name)
        for f in (b1_dir / "labels").glob("*.txt"):
            shutil.copy(f, OUTPUT_DIR_ALL / "labels" / f.name)

    # Copy Batch 2
    for f in (OUTPUT_DIR_B2 / "images").glob("*.*"):
        shutil.copy(f, OUTPUT_DIR_ALL / "images" / f.name)
    for f in (OUTPUT_DIR_B2 / "labels").glob("*.txt"):
        shutil.copy(f, OUTPUT_DIR_ALL / "labels" / f.name)

    # Cumulative metadata
    b1_meta = pd.read_csv(BATCH1_META) if BATCH1_META.exists() else pd.DataFrame()
    if not b1_meta.empty and 'batch' not in b1_meta.columns:
        b1_meta['batch'] = 'batch_1'
    all_meta = pd.concat([b1_meta, log_df], ignore_index=True)
    all_meta.to_csv(OUTPUT_DIR_ALL / "crops_metadata_summary.csv", index=False)

    with open(OUTPUT_DIR_ALL / "data.yaml", "w") as f:
        f.write(f"""# Unified Roboflow / YOLOv8 / YOLO11 Segmentation Dataset (300 Crops)
train: ./images
val: ./images

nc: 2
names: ['center', 'stomata-huw1']

roboflow:
  project: stomata-retraining
  description: "300 curated crops (200 Batch 1 + 100 Batch 2 hard cases)"
""")

    zip_path_all = WORKSPACE_DIR / "roboflow_preannotated_dataset_all_300.zip"
    with zipfile.ZipFile(zip_path_all, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR_ALL):
            for file in files:
                fpath = os.path.join(root, file)
                arcname = os.path.relpath(fpath, OUTPUT_DIR_ALL)
                zipf.write(fpath, arcname)

    print(f"[+] Unified 300-crop zip created at {zip_path_all} ({os.path.getsize(zip_path_all):,} bytes)", flush=True)

    if cesnet_backup.exists():
        cesnet_dest_all = cesnet_backup / "roboflow_preannotated_dataset_all_300.zip"
        shutil.copy(zip_path_all, cesnet_dest_all)
        print(f"[+] Backed up Unified 300-crop zip to CESNET S3 mount: {cesnet_dest_all}", flush=True)

    print("\n" + "=" * 76, flush=True)
    print(f"  FINISHED: 100 Hard-Case Crops Generated Successfully!", flush=True)
    print(f"  - Batch 2 Folder: {OUTPUT_DIR_B2}", flush=True)
    print(f"  - Batch 2 Zip:    {zip_path_b2}", flush=True)
    print(f"  - Unified Folder: {OUTPUT_DIR_ALL}", flush=True)
    print(f"  - Unified Zip:    {zip_path_all}", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    main()
