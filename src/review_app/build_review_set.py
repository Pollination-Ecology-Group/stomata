#!/usr/bin/env python3
"""
build_review_set.py
Prepares a balanced, stratified QC review set for the Stomata Measurement Precision App.

Selects representative stomata across cytotypes (nat_2x, ory_2x, neo_4x, nat_4x),
leaf sides (abaxial vs adaxial), focus qualities, and aperture widths.
Downloads the source micrographs from CESNET S3, extracts high-res 360x360 stoma crops,
computes geometric vector coordinates for canvas rendering, and uploads crops & manifest to S3.
"""

import os
import sys
import json
import base64
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import boto3
from botocore.config import Config

PX_TO_UM = 50.0 / 451.3958
CROP_SIZE = 360
REVIEW_S3_PREFIX = "Ostatni/Stomata/Review"
BUCKET = "peg-bucket"


def get_s3_client():
    # Attempt local env vars first
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")

    if not ak or not sk:
        # Fetch from k8s secret
        try:
            out = subprocess.check_output(
                ["kubectl", "get", "secret", "s3-credentials", "-n", "stenc-ns", "-o", "json"]
            )
            data = json.loads(out)["data"]
            ak = base64.b64decode(data["access-key"]).decode().strip()
            sk = base64.b64decode(data["secret-key"]).decode().strip()
        except Exception as e:
            print(f"[!] Warning: Could not read k8s s3 secret: {e}")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"})
    )


def compute_geometry_vectors(row, local_cx, local_cy):
    """
    Computes exact 2D line endpoints and ticks for major axis, minor axis,
    pore aperture, and guard cell thickness within the local crop coordinate space.
    """
    major_px = float(row["stoma_major_um"]) / PX_TO_UM
    minor_px = float(row["stoma_minor_um"]) / PX_TO_UM
    pore_px = float(row["pore_minor_um"]) / PX_TO_UM
    angle_deg = float(row.get("angle_deg", 0.0))

    # In extract_traits.py:
    # angle_deg was saved as angle_s (minAreaRect angle).
    # If w < h: minor_angle = angle_s, major_angle = angle_s + 90
    # If w >= h: minor_angle = angle_s - 90, major_angle = angle_s
    # In OpenCV minAreaRect, minor axis is the shorter dimension.
    # To be consistent with extract_traits.py:
    minor_rad = np.deg2rad(angle_deg)
    # Major axis is perpendicular to minor axis
    major_rad = minor_rad + np.pi / 2.0

    # Unit vectors
    u_min = np.array([np.cos(minor_rad), np.sin(minor_rad)])
    u_maj = np.array([np.cos(major_rad), np.sin(major_rad)])

    center = np.array([local_cx, local_cy])

    # 1. Major Axis endpoints (stoma full length)
    maj_half = major_px / 2.0
    p_maj1 = center - maj_half * u_maj
    p_maj2 = center + maj_half * u_maj

    # 2. Minor Axis endpoints (stoma full width)
    min_half = minor_px / 2.0
    p_min1 = center - min_half * u_min
    p_min2 = center + min_half * u_min

    # 3. Pore Aperture endpoints (central opening)
    pore_half = pore_px / 2.0
    p_pore1 = center - pore_half * u_min
    p_pore2 = center + pore_half * u_min

    # 4. Guard cell thickness intervals along minor axis:
    # Guard cell 1: from p_min1 to p_pore1
    # Guard cell 2: from p_pore2 to p_min2
    gc_w_px = (minor_px - pore_px) / 2.0

    # Perpendicular ticks for visual caliper alignment
    tick_len = 8.0
    def tick_segment(pt, u_perp):
        return [
            [round(float(pt[0] - tick_len * u_perp[0]), 1), round(float(pt[1] - tick_len * u_perp[1]), 1)],
            [round(float(pt[0] + tick_len * u_perp[0]), 1), round(float(pt[1] + tick_len * u_perp[1]), 1)]
        ]

    return {
        "major_line": [
            [round(float(p_maj1[0]), 1), round(float(p_maj1[1]), 1)],
            [round(float(p_maj2[0]), 1), round(float(p_maj2[1]), 1)]
        ],
        "minor_line": [
            [round(float(p_min1[0]), 1), round(float(p_min1[1]), 1)],
            [round(float(p_min2[0]), 1), round(float(p_min2[1]), 1)]
        ],
        "pore_line": [
            [round(float(p_pore1[0]), 1), round(float(p_pore1[1]), 1)],
            [round(float(p_pore2[0]), 1), round(float(p_pore2[1]), 1)]
        ],
        "guard_cell_ticks": [
            tick_segment(p_min1, u_maj),
            tick_segment(p_pore1, u_maj),
            tick_segment(p_pore2, u_maj),
            tick_segment(p_min2, u_maj)
        ],
        "major_ticks": [
            tick_segment(p_maj1, u_min),
            tick_segment(p_maj2, u_min)
        ],
        "stoma_ellipse": {
            "cx": round(float(local_cx), 1),
            "cy": round(float(local_cy), 1),
            "rx": round(float(maj_half), 1),
            "ry": round(float(min_half), 1),
            "angle_deg": round(float(np.degrees(major_rad)), 1)
        },
        "pore_ellipse": {
            "cx": round(float(local_cx), 1),
            "cy": round(float(local_cy), 1),
            "rx": round(float(float(row.get("pore_major_um", row["stoma_major_um"] * 0.5)) / PX_TO_UM / 2.0), 1),
            "ry": round(float(pore_half), 1),
            "angle_deg": round(float(np.degrees(major_rad)), 1)
        }
    }


def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / "data"
    stomata_csv_path = data_dir / "stomata_traits_all_3253_stomata.csv"
    summary_csv_path = data_dir / "stomata_traits_all_3253_image_summary.csv"

    out_dir = root_dir / "src" / "review_app"
    crops_dir = out_dir / "static" / "crops"
    overviews_dir = out_dir / "static" / "overviews"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overviews_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading traits from {stomata_csv_path}...")
    df_stoma = pd.read_csv(stomata_csv_path, low_memory=False)
    df_summary = pd.read_csv(summary_csv_path, low_memory=False) if summary_csv_path.exists() else None

    # Target images: diverse set spanning all 4 cytotypes and both leaf sides
    # 1. Start with the 10 images analyzed in test_detection_sample
    seed_images = [
        "2-4-C_AD_3.png", "5-10-D_AB_3.png", "5-10-D_AB_4.png", "5-10-D_AD_3.png",
        "5-9-D_AB_3.png", "5-9-E_AB_1.png", "5-9-E_AD_2.png", "6-3-C_AB_3.png",
        "6-3-D_AB_2.png", "6-4-A_AD_1.png"
    ]
    # 2. Add representative ory_2x images (both AB and AD)
    ory_imgs = df_stoma[df_stoma["cytotype"] == "ory_2x"]["image_name"].drop_duplicates()
    ory_ab = [img for img in ory_imgs if "_AB_" in img][:2]
    ory_ad = [img for img in ory_imgs if "_AD_" in img][:2]

    # 3. Add nat_4x images
    nat4_imgs = df_stoma[df_stoma["cytotype"] == "nat_4x"]["image_name"].drop_duplicates()
    nat4_ab = [img for img in nat4_imgs if "_AB_" in img and img not in seed_images][:2]
    nat4_ad = [img for img in nat4_imgs if "_AD_" in img and img not in seed_images][:2]

    selected_images = list(dict.fromkeys(seed_images + ory_ab + ory_ad + nat4_ab + nat4_ad))
    print(f"[*] Selected {len(selected_images)} representative micrographs:")
    for img_name in selected_images:
        sub = df_stoma[df_stoma["image_name"] == img_name]
        cyto = sub["cytotype"].iloc[0] if len(sub) else "unknown"
        side = sub["leaf_side"].iloc[0] if len(sub) else "unknown"
        print(f"    - {img_name:<18} [{cyto:<7}, {side:<7}] ({len(sub)} stomata)")

    s3 = get_s3_client()

    review_items = []
    slides_metadata = []

    for img_idx, img_name in enumerate(selected_images, 1):
        print(f"\n[{img_idx}/{len(selected_images)}] Processing {img_name}...")
        sub_df = df_stoma[df_stoma["image_name"] == img_name].copy()
        if len(sub_df) == 0:
            continue

        stem = Path(img_name).stem
        overview_filename = f"{stem}_overview.jpg"
        overview_local = overviews_dir / overview_filename
        overview_s3_key = f"{REVIEW_S3_PREFIX}/overviews/{overview_filename}"

        # Check if source image is available locally or on S3
        img_bgr = None
        # Check local test dirs
        for local_dir in [
            root_dir / "test_detection_results" / "annotated_images",
            Path("/home/meow/cesnet_new/Ostatni/Stomata/detect/stomata_trait_comparison_AA012-20251216T171633Z-3-002/stomata_trait_comparison_AA012")
        ]:
            candidate = local_dir / img_name
            if candidate.exists():
                img_bgr = cv2.imread(str(candidate))
                if img_bgr is not None:
                    print(f"    Loaded from local: {candidate}")
                    break

        if img_bgr is None:
            # Download from S3
            s3_key = f"Ostatni/Stomata/extracted/stomata_trait_comparison/{img_name}"
            print(f"    Downloading from s3://{BUCKET}/{s3_key}...")
            try:
                t0 = time.time()
                obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
                buf = obj["Body"].read()
                img_bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                print(f"    Downloaded in {time.time() - t0:.2f}s ({img_bgr.shape[1]}x{img_bgr.shape[0]})")
            except Exception as e:
                print(f"    [-] Failed to download {s3_key}: {e}")
                continue

        H_img, W_img = img_bgr.shape[:2]

        # Generate scaled overview image for slide browser (width = 1600 px)
        scale_overview = 1600.0 / W_img
        new_w = 1600
        new_h = int(round(H_img * scale_overview))
        overview_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(overview_local), overview_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Stratified selection of stomata on this slide for the precision review queue:
        # - Ploidy suitable (in focus, fully interior)
        # - Lower focus / blurry (to test robustness)
        # - Small / closed aperture (subpixel groove)
        # - Wide open aperture
        suitable = sub_df[sub_df["is_suitable_for_ploidy_sizing"] == True]
        unsuitable = sub_df[sub_df["is_suitable_for_ploidy_sizing"] == False]

        chosen_indices = []
        if len(suitable) > 0:
            # Sort by focus score and pick top 2
            top_focus = suitable.sort_values(by="focus_score", ascending=False).head(2)
            chosen_indices.extend(top_focus.index.tolist())

            # Pick largest & smallest aperture among suitable
            if len(suitable) >= 4:
                wide = suitable.sort_values(by="aperture_ratio", ascending=False).head(2)
                narrow = suitable.sort_values(by="aperture_ratio", ascending=True).head(2)
                chosen_indices.extend(wide.index.tolist())
                chosen_indices.extend(narrow.index.tolist())

        if len(unsuitable) > 0:
            # Pick a borderline or low focus stoma
            low_f = unsuitable.sort_values(by="focus_score", ascending=True).head(2)
            chosen_indices.extend(low_f.index.tolist())

        chosen_indices = list(dict.fromkeys(chosen_indices))[:8]  # up to 8 per slide
        if len(chosen_indices) < 5:
            # Add remaining to reach at least 5
            remaining = [idx for idx in sub_df.index if idx not in chosen_indices]
            chosen_indices.extend(remaining[: 5 - len(chosen_indices)])

        # Slide-level stoma pins (for full interactive micrograph map)
        slide_pins = []
        for _, s_row in sub_df.iterrows():
            slide_pins.append({
                "stoma_idx": int(s_row["stoma_idx"]),
                "cx": round(float(s_row["centroid_x"]), 1),
                "cy": round(float(s_row["centroid_y"]), 1),
                "norm_x": round(float(s_row["centroid_x"]) / W_img, 4),
                "norm_y": round(float(s_row["centroid_y"]) / H_img, 4),
                "major_axis_um": float(s_row["stoma_major_um"]),
                "minor_axis_um": float(s_row["stoma_minor_um"]),
                "pore_width_um": float(s_row["pore_minor_um"]),
                "guard_cell_width_um": float(s_row["guard_cell_width_um"]),
                "focus_score": float(s_row["focus_score"]),
                "is_suitable_for_ploidy_sizing": bool(s_row["is_suitable_for_ploidy_sizing"]),
                "counts_for_density": bool(s_row["counts_for_density"]),
            })

        slides_metadata.append({
            "image_name": img_name,
            "stem": stem,
            "cytotype": str(sub_df["cytotype"].iloc[0]),
            "leaf_side": str(sub_df["leaf_side"].iloc[0]),
            "population": str(sub_df["population"].iloc[0]),
            "plant_ID": str(sub_df["plant_ID"].iloc[0]),
            "ploidy": float(sub_df["ploidy"].iloc[0]),
            "width": W_img,
            "height": H_img,
            "total_stomata": len(sub_df),
            "suitable_stomata": int((sub_df["is_suitable_for_ploidy_sizing"] == True).sum()),
            "density_per_mm2": round(len(sub_df) / ((W_img * PX_TO_UM / 1000.0) * (H_img * PX_TO_UM / 1000.0)), 1),
            "overview_url": f"static/overviews/{overview_filename}",
            "overview_s3_key": overview_s3_key,
            "stomata_pins": slide_pins
        })

        # Extract high-res crops for queued stomata
        half_crop = CROP_SIZE // 2
        for s_idx in chosen_indices:
            row = sub_df.loc[s_idx]
            cx = float(row["centroid_x"])
            cy = float(row["centroid_y"])

            x1 = int(round(cx - half_crop))
            y1 = int(round(cy - half_crop))
            x2 = x1 + CROP_SIZE
            y2 = y1 + CROP_SIZE

            # Extract crop with border replication if close to edge
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - W_img)
            pad_bottom = max(0, y2 - H_img)

            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(W_img, x2)
            src_y2 = min(H_img, y2)

            crop_patch = img_bgr[src_y1:src_y2, src_x1:src_x2]
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                crop_patch = cv2.copyMakeBorder(
                    crop_patch, pad_top, pad_bottom, pad_left, pad_right,
                    borderType=cv2.BORDER_REFLECT_101
                )

            # Local coordinates inside crop
            local_cx = cx - x1
            local_cy = cy - y1

            item_id = f"{stem}_s{int(row['stoma_idx']):03d}"
            crop_filename = f"{item_id}.jpg"
            crop_local = crops_dir / crop_filename
            crop_s3_key = f"{REVIEW_S3_PREFIX}/crops/{crop_filename}"

            cv2.imwrite(str(crop_local), crop_patch, [cv2.IMWRITE_JPEG_QUALITY, 92])

            vectors = compute_geometry_vectors(row, local_cx, local_cy)

            review_items.append({
                "id": item_id,
                "image_name": img_name,
                "stoma_idx": int(row["stoma_idx"]),
                "plant_ID": str(row["plant_ID"]),
                "population": str(row["population"]),
                "cytotype": str(row["cytotype"]),
                "ploidy": float(row["ploidy"]),
                "leaf_side": str(row["leaf_side"]),
                "side_code": str(row["side_code"]),
                "crop_url": f"static/crops/{crop_filename}",
                "crop_s3_key": crop_s3_key,
                "overview_url": f"static/overviews/{overview_filename}",
                "overview_s3_key": overview_s3_key,
                "crop_box": [int(x1), int(y1), int(x2), int(y2)],
                "crop_w": CROP_SIZE,
                "crop_h": CROP_SIZE,
                "centroid_x": round(cx, 1),
                "centroid_y": round(cy, 1),
                "local_cx": round(local_cx, 1),
                "local_cy": round(local_cy, 1),
                "major_axis_um": float(row["stoma_major_um"]),
                "minor_axis_um": float(row["stoma_minor_um"]),
                "pore_width_um": float(row["pore_minor_um"]),
                "guard_cell_width_um": float(row["guard_cell_width_um"]),
                "aspect_ratio": float(row["stoma_aspect_ratio"]),
                "aperture_ratio": float(row.get("aperture_ratio", 0.0)),
                "focus_score": float(row["focus_score"]),
                "stoma_conf": float(row["stoma_conf"]),
                "center_conf": float(row["center_conf"]),
                "is_well_focused": bool(row["is_well_focused"]),
                "is_suitable_for_ploidy_sizing": bool(row["is_suitable_for_ploidy_sizing"]),
                "counts_for_density": bool(row["counts_for_density"]),
                "stage2_refined": bool(row.get("stage2_refined", False)),
                "vectors": vectors
            })

    # Global benchmark statistics
    global_stats = {
        "total_stomata_measured": len(df_stoma),
        "total_micrographs": int(df_stoma["image_name"].nunique()),
        "ploidy_suitable_count": int((df_stoma["is_suitable_for_ploidy_sizing"] == True).sum()),
        "ploidy_suitable_pct": round(float((df_stoma["is_suitable_for_ploidy_sizing"] == True).mean()) * 100, 1),
        "cytotype_medians": df_stoma.groupby("cytotype")["stoma_major_um"].median().round(2).to_dict(),
        "guard_cell_medians": df_stoma.groupby("cytotype")["guard_cell_width_um"].median().round(2).to_dict(),
        "pore_medians": df_stoma.groupby("cytotype")["pore_minor_um"].median().round(2).to_dict(),
        "px_to_um": PX_TO_UM
    }

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "px_to_um": PX_TO_UM,
        "global_stats": global_stats,
        "items": review_items,
        "slides": slides_metadata
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Manifest generated with {len(review_items)} stoma crops and {len(slides_metadata)} slides.")
    print(f"    Saved locally to {manifest_path}")

    # Upload manifest and crops to S3
    print(f"[*] Uploading crops and manifest to S3 (s3://{BUCKET}/{REVIEW_S3_PREFIX}/)...")
    s3.upload_file(str(manifest_path), BUCKET, f"{REVIEW_S3_PREFIX}/manifest.json")
    print(f"    Uploaded s3://{BUCKET}/{REVIEW_S3_PREFIX}/manifest.json")

    for item in review_items:
        local_crop = out_dir / item["crop_url"]
        if local_crop.exists():
            s3.upload_file(str(local_crop), BUCKET, item["crop_s3_key"])

    for slide in slides_metadata:
        local_ov = out_dir / slide["overview_url"]
        if local_ov.exists():
            s3.upload_file(str(local_ov), BUCKET, slide["overview_s3_key"])

    print(f"[+] All {len(review_items)} crops and {len(slides_metadata)} overviews uploaded to S3 successfully!")


if __name__ == "__main__":
    main()
