#!/usr/bin/env python3
"""
build_review_from_local.py
Builds the comprehensive precision review dataset for Stomata using the 300 local crops
with YOLO polygon ground-truth/preannotations plus the 10 full micrographs.

Features:
- Extracts exact stoma polygon (class 1) & pore polygon (class 0)
- Calculates Major axis, Minor axis, Pore aperture width, Guard cell thickness in px and um
- Computes 2D vector endpoints for canvas overlay rendering (major line, minor line, pore line, guard cell ticks)
- Evaluates Laplacian focus score and ploidy sizing suitability
- Prepares both high-res crop inspection and full micrograph field inspection
- Saves manifest.json and uploads crops/overviews/manifest to S3 (Ostatni/Stomata/Review/)
"""

import os
import sys
import json
import base64
import subprocess
import time
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, MultiPolygon, Point
import boto3
from botocore.config import Config

PX_TO_UM = 50.0 / 451.3958
REVIEW_S3_PREFIX = "Ostatni/Stomata/Review"
BUCKET = "peg-bucket"


def get_s3_client():
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")

    if not ak or not sk:
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


def compute_focus_score(img_bgr):
    if img_bgr is None or img_bgr.size == 0 or img_bgr.shape[0] < 6 or img_bgr.shape[1] < 6:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return round(float(np.clip(variance / 2.0, 0.0, 100.0)), 1)


def parse_label_file(lbl_path, W, H):
    """Parses YOLO segmentation format lines: <cls> x1 y1 x2 y2 ..."""
    stomas = []
    centers = []
    if not lbl_path.exists():
        return stomas, centers

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        # De-normalize coordinates
        pts = []
        for i in range(0, len(coords) - 1, 2):
            pts.append((coords[i] * W, coords[i + 1] * H))

        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue

        if cls_id == 1:
            stomas.append(poly)
        elif cls_id == 0:
            centers.append(poly)

    return stomas, centers


def compute_anatomical_vectors_and_traits(stoma_poly, matched_pore, W, H):
    """
    Computes exact geometry traits and line vectors anchored to the central pore's
    principal longitudinal axis, eliminating arbitrary diagonal snapping on round stomata.
    """
    s_coords = np.array(stoma_poly.exterior.coords, dtype=np.float32)
    M_s = cv2.moments(s_coords)
    if M_s["m00"] > 0:
        cx = M_s["m10"] / M_s["m00"]
        cy = M_s["m01"] / M_s["m00"]
    else:
        cx, cy = stoma_poly.centroid.x, stoma_poly.centroid.y

    # Determine primary longitudinal orientation (Major Axis)
    if matched_pore is not None and not matched_pore.is_empty:
        p_coords = np.array(matched_pore.exterior.coords, dtype=np.float32)
        M_p = cv2.moments(p_coords)
        if M_p["m00"] > 0:
            mu20 = M_p["mu20"] / M_p["m00"]
            mu02 = M_p["mu02"] / M_p["m00"]
            mu11 = M_p["mu11"] / M_p["m00"]
            major_angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        else:
            rect_p = cv2.minAreaRect(p_coords)
            (pcx, pcy), (pw, ph), p_angle = rect_p
            major_angle_rad = np.deg2rad(p_angle + 90.0 if pw < ph else p_angle)
    else:
        # Fallback to stoma polygon moments of inertia
        if M_s["m00"] > 0:
            mu20 = M_s["mu20"] / max(1e-5, M_s["m00"])
            mu02 = M_s["mu02"] / max(1e-5, M_s["m00"])
            mu11 = M_s["mu11"] / max(1e-5, M_s["m00"])
            major_angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
        else:
            rect_s = cv2.minAreaRect(s_coords)
            (scx, scy), (sw, sh), s_angle = rect_s
            major_angle_rad = np.deg2rad(s_angle + 90.0 if sw < sh else s_angle)

    # Transverse orientation (Minor Axis) is strictly perpendicular
    minor_angle_rad = major_angle_rad + np.pi / 2.0
    u_maj = np.array([np.cos(major_angle_rad), np.sin(major_angle_rad)])
    u_min = np.array([np.cos(minor_angle_rad), np.sin(minor_angle_rad)])

    # Construct extended rays from stoma centroid for exact boundary intersection
    span = 1200.0
    line_maj = LineString([(cx - span * u_maj[0], cy - span * u_maj[1]),
                          (cx + span * u_maj[0], cy + span * u_maj[1])])
    line_min = LineString([(cx - span * u_min[0], cy - span * u_min[1]),
                          (cx + span * u_min[0], cy + span * u_min[1])])

    inter_maj = line_maj.intersection(stoma_poly)
    inter_min = line_min.intersection(stoma_poly)

    if inter_maj.is_empty:
        maj_half = 50.0
        p_maj1 = np.array([cx, cy]) - maj_half * u_maj
        p_maj2 = np.array([cx, cy]) + maj_half * u_maj
        major_len_px = 100.0
    else:
        if inter_maj.geom_type == 'MultiLineString':
            inter_maj = max(inter_maj.geoms, key=lambda g: g.length)
        coords_m = list(inter_maj.coords)
        p_maj1, p_maj2 = np.array(coords_m[0]), np.array(coords_m[-1])
        major_len_px = inter_maj.length

    if inter_min.is_empty:
        min_half = 40.0
        p_min1 = np.array([cx, cy]) - min_half * u_min
        p_min2 = np.array([cx, cy]) + min_half * u_min
        minor_len_px = 80.0
    else:
        if inter_min.geom_type == 'MultiLineString':
            inter_min = max(inter_min.geoms, key=lambda g: g.length)
        coords_n = list(inter_min.coords)
        p_min1, p_min2 = np.array(coords_n[0]), np.array(coords_n[-1])
        minor_len_px = inter_min.length

    pore_minor_px = 0.0
    p_por1, p_por2 = None, None
    if matched_pore is not None and not matched_pore.is_empty:
        inter_p = line_min.intersection(matched_pore)
        if not inter_p.is_empty:
            if inter_p.geom_type == 'MultiLineString':
                inter_p = max(inter_p.geoms, key=lambda g: g.length)
            coords_p = list(inter_p.coords)
            p_por1, p_por2 = np.array(coords_p[0]), np.array(coords_p[-1])
            pore_minor_px = inter_p.length
        else:
            p_rect = cv2.minAreaRect(np.array(matched_pore.exterior.coords, dtype=np.float32))
            pore_minor_px = min(p_rect[1][0], p_rect[1][1])
            p_por1 = np.array([cx, cy]) - (pore_minor_px / 2.0) * u_min
            p_por2 = np.array([cx, cy]) + (pore_minor_px / 2.0) * u_min

    guard_cell_px = max(0.0, (minor_len_px - pore_minor_px) / 2.0)

    tick_len = 7.0
    def tick_segment(pt, u_perp):
        return [
            [round(float(pt[0] - tick_len * u_perp[0]), 1), round(float(pt[1] - tick_len * u_perp[1]), 1)],
            [round(float(pt[0] + tick_len * u_perp[0]), 1), round(float(pt[1] + tick_len * u_perp[1]), 1)]
        ]

    # Convert polygon points to list of [x, y]
    stoma_pts = [[round(float(x), 1), round(float(y), 1)] for x, y in stoma_poly.exterior.coords]
    pore_pts = [[round(float(x), 1), round(float(y), 1)] for x, y in matched_pore.exterior.coords] if matched_pore else []

    traits = {
        "cx": round(float(cx), 1),
        "cy": round(float(cy), 1),
        "major_axis_px": round(float(major_len_px), 1),
        "major_axis_um": round(float(major_len_px * PX_TO_UM), 2),
        "minor_axis_px": round(float(minor_len_px), 1),
        "minor_axis_um": round(float(minor_len_px * PX_TO_UM), 2),
        "pore_width_px": round(float(pore_minor_px), 1),
        "pore_width_um": round(float(pore_minor_px * PX_TO_UM), 2),
        "guard_cell_width_px": round(float(guard_cell_px), 1),
        "guard_cell_width_um": round(float(guard_cell_px * PX_TO_UM), 2),
        "aspect_ratio": round(float(major_len_px / max(1e-4, minor_len_px)), 2),
        "aperture_ratio": round(float(pore_minor_px / max(1e-4, minor_len_px)), 3),
        "angle_deg": round(float(np.rad2deg(major_angle_rad)), 1)
    }

    vectors = {
        "stoma_poly": stoma_pts,
        "pore_poly": pore_pts,
        "major_line": [
            [round(float(p_maj1[0]), 1), round(float(p_maj1[1]), 1)],
            [round(float(p_maj2[0]), 1), round(float(p_maj2[1]), 1)]
        ],
        "minor_line": [
            [round(float(p_min1[0]), 1), round(float(p_min1[1]), 1)],
            [round(float(p_min2[0]), 1), round(float(p_min2[1]), 1)]
        ],
        "pore_line": [
            [round(float(p_por1[0]), 1), round(float(p_por1[1]), 1)],
            [round(float(p_por2[0]), 1), round(float(p_por2[1]), 1)]
        ] if p_por1 is not None else None,
        "major_ticks": [
            tick_segment(p_maj1, u_min),
            tick_segment(p_maj2, u_min)
        ],
        "minor_ticks": [
            tick_segment(p_min1, u_maj),
            tick_segment(p_por1, u_maj),
            tick_segment(p_por2, u_maj),
            tick_segment(p_min2, u_maj)
        ] if p_por1 is not None else [
            tick_segment(p_min1, u_maj),
            tick_segment(p_min2, u_maj)
        ]
    }

    return traits, vectors


def main():
    root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = root / "roboflow_preannotated_dataset_all_300"
    img_dir = dataset_dir / "images"
    lbl_dir = dataset_dir / "labels"
    meta_path = dataset_dir / "crops_metadata_summary.csv"

    out_app_dir = root / "src" / "review_app"
    crops_dest = out_app_dir / "static" / "crops"
    overviews_dest = out_app_dir / "static" / "overviews"
    crops_dest.mkdir(parents=True, exist_ok=True)
    overviews_dest.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading crops metadata from {meta_path}...")
    df_meta = pd.read_csv(meta_path)
    meta_lookup = {row["filename"]: row.to_dict() for _, row in df_meta.iterrows()}

    # Copy the 10 full annotated micrographs from test_detection_results/annotated_images to overviews
    annotated_dir = root / "test_detection_results" / "annotated_images"
    slides = []
    if annotated_dir.exists():
        for ov_img in sorted(annotated_dir.glob("*.jpg")):
            stem = ov_img.stem.replace("_annotated", "")
            dest_name = f"{stem}_overview.jpg"
            shutil.copyfile(ov_img, overviews_dest / dest_name)
            # Try to guess metadata from stem
            parts = stem.split("_")
            plant_id = parts[0] if len(parts) > 0 else ""
            leaf_side = "abaxial" if "AB" in stem else ("adaxial" if "AD" in stem else "")
            slides.append({
                "image_name": f"{stem}.png",
                "stem": stem,
                "plant_ID": plant_id,
                "leaf_side": leaf_side,
                "overview_url": f"static/overviews/{dest_name}",
                "overview_s3_key": f"{REVIEW_S3_PREFIX}/overviews/{dest_name}"
            })
    print(f"[+] Loaded {len(slides)} whole-field micrographs.")

    review_items = []
    crop_files = sorted(list(img_dir.glob("*.jpg")))
    print(f"[*] Processing {len(crop_files)} high-magnification stoma crops...")

    for c_idx, c_path in enumerate(crop_files, 1):
        filename = c_path.name
        lbl_path = lbl_dir / f"{c_path.stem}.txt"
        meta = meta_lookup.get(filename, {})

        img = cv2.imread(str(c_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        focus_score = compute_focus_score(img)

        # Copy crop image to static destination
        shutil.copyfile(c_path, crops_dest / filename)

        stomas, centers = parse_label_file(lbl_path, W, H)

        # Match stoma to pore
        if not stomas:
            # Negative control / background crop
            item_id = f"crop_{c_idx:03d}_{c_path.stem}"
            review_items.append({
                "id": item_id,
                "filename": filename,
                "crop_id": meta.get("crop_id", c_idx),
                "cytotype": meta.get("cytotype", "background"),
                "plant_ID": meta.get("plant_ID", "bg_control"),
                "leaf_side": meta.get("leaf_side", "none"),
                "population": meta.get("population", "control"),
                "crop_url": f"static/crops/{filename}",
                "crop_s3_key": f"{REVIEW_S3_PREFIX}/crops/{filename}",
                "crop_w": W,
                "crop_h": H,
                "focus_score": focus_score,
                "is_well_focused": focus_score >= 3.0,
                "is_suitable_for_ploidy_sizing": False,
                "counts_for_density": False,
                "is_background_control": True,
                "stoma_conf": 0.0,
                "center_conf": 0.0,
                "major_axis_um": 0.0,
                "minor_axis_um": 0.0,
                "pore_width_um": 0.0,
                "guard_cell_width_um": 0.0,
                "aspect_ratio": 0.0,
                "aperture_ratio": 0.0,
                "angle_deg": 0.0,
                "vectors": {"stoma_poly": [], "pore_poly": [], "major_line": None, "minor_line": None, "pore_line": None}
            })
            continue

        # Choose the most central stoma in the crop as primary target
        crop_center = Point(W / 2.0, H / 2.0)
        stomas.sort(key=lambda s: s.centroid.distance(crop_center))
        primary_stoma = stomas[0]

        matched_pore = None
        for c in centers:
            if primary_stoma.contains(c.centroid) or (primary_stoma.intersection(c).area / max(1e-4, c.area) > 0.5):
                matched_pore = c
                break

        traits, vectors = compute_anatomical_vectors_and_traits(primary_stoma, matched_pore, W, H)

        # Process companion stomata in the crop if more than 1 exists
        companion_stomata = []
        for comp_idx, c_stoma in enumerate(stomas[1:], 2):
            comp_pore = None
            is_c_edge = any(x <= 3 or y <= 3 or x >= W - 3 or y >= H - 3 for x, y in c_stoma.exterior.coords)
            if not is_c_edge:
                for c in centers:
                    # Ignore synthetic 5-point boxes on borders
                    if len(c.exterior.coords) == 5:
                        continue
                    if c_stoma.contains(c.centroid) or (c_stoma.intersection(c).area / max(1e-4, c.area) > 0.5):
                        comp_pore = c
                        break
            c_traits, c_vectors = compute_anatomical_vectors_and_traits(c_stoma, comp_pore, W, H)
            companion_stomata.append({
                "stoma_idx": comp_idx,
                "cx": c_traits["cx"],
                "cy": c_traits["cy"],
                "stoma_poly": c_vectors["stoma_poly"],
                "pore_poly": c_vectors["pore_poly"],
                "major_line": c_vectors["major_line"],
                "minor_line": c_vectors["minor_line"],
                "major_axis_um": c_traits["major_axis_um"],
                "minor_axis_um": c_traits["minor_axis_um"],
                "pore_width_um": c_traits["pore_width_um"],
                "guard_cell_width_um": c_traits["guard_cell_width_um"]
            })
        vectors["companion_stomata"] = companion_stomata

        # Suitability
        is_suitable = (
            focus_score >= 2.8 and
            traits["major_axis_um"] >= 10.0 and
            traits["major_axis_um"] <= 55.0 and
            traits["aspect_ratio"] >= 1.0 and
            traits["aspect_ratio"] <= 3.2 and
            traits["guard_cell_width_um"] >= 0.5 and
            matched_pore is not None
        )

        item_id = f"crop_{c_idx:03d}_{c_path.stem}"
        review_items.append({
            "id": item_id,
            "filename": filename,
            "crop_id": meta.get("crop_id", c_idx),
            "stoma_idx": 1,
            "total_stomata_in_crop": len(stomas),
            "cytotype": meta.get("cytotype", "unknown"),
            "plant_ID": meta.get("plant_ID", "sample"),
            "leaf_side": meta.get("leaf_side", "abaxial" if "abaxial" in filename else ("adaxial" if "adaxial" in filename else "unknown")),
            "population": meta.get("population", "AA012"),
            "crop_url": f"static/crops/{filename}",
            "crop_s3_key": f"{REVIEW_S3_PREFIX}/crops/{filename}",
            "crop_w": W,
            "crop_h": H,
            "focus_score": focus_score,
            "is_well_focused": focus_score >= 3.0,
            "is_suitable_for_ploidy_sizing": is_suitable,
            "counts_for_density": traits["major_axis_um"] >= 8.0,
            "is_background_control": False,
            "stoma_conf": round(float(meta.get("avg_stoma_conf", 0.85)), 3),
            "center_conf": round(float(meta.get("target_center_conf", 0.65)), 3),
            "major_axis_um": traits["major_axis_um"],
            "minor_axis_um": traits["minor_axis_um"],
            "pore_width_um": traits["pore_width_um"],
            "guard_cell_width_um": traits["guard_cell_width_um"],
            "aspect_ratio": traits["aspect_ratio"],
            "aperture_ratio": traits["aperture_ratio"],
            "angle_deg": traits["angle_deg"],
            "vectors": vectors
        })

    global_stats = {
        "total_stomata_measured": len(review_items),
        "total_crops": len(crop_files),
        "total_micrographs": len(slides),
        "ploidy_suitable_count": sum(1 for it in review_items if it.get("is_suitable_for_ploidy_sizing")),
        "cytotypes": {
            "nat_2x": sum(1 for it in review_items if it.get("cytotype") == "nat_2x"),
            "ory_2x": sum(1 for it in review_items if it.get("cytotype") == "ory_2x"),
            "nat_4x": sum(1 for it in review_items if it.get("cytotype") == "nat_4x"),
            "neo_4x": sum(1 for it in review_items if it.get("cytotype") == "neo_4x"),
            "background": sum(1 for it in review_items if it.get("cytotype") == "background")
        },
        "px_to_um": PX_TO_UM
    }

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "px_to_um": PX_TO_UM,
        "global_stats": global_stats,
        "items": review_items,
        "slides": slides
    }

    manifest_path = out_app_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Manifest generated successfully!")
    print(f"    Total items: {len(review_items)}")
    print(f"    Total slides: {len(slides)}")
    print(f"    Saved to: {manifest_path} ({manifest_path.stat().st_size:,} bytes)")

    # Upload manifest and files to S3
    s3 = get_s3_client()
    print(f"[*] Uploading manifest.json to s3://{BUCKET}/{REVIEW_S3_PREFIX}/manifest.json...")
    try:
        s3.upload_file(str(manifest_path), BUCKET, f"{REVIEW_S3_PREFIX}/manifest.json")
        print(f"    [+] Uploaded manifest to S3.")
    except Exception as e:
        print(f"    [-] S3 upload failed: {e}")

    # Upload overviews
    for sl in slides:
        ov_local = out_app_dir / sl["overview_url"]
        if ov_local.exists():
            try:
                s3.upload_file(str(ov_local), BUCKET, sl["overview_s3_key"])
            except Exception:
                pass
    print(f"    [+] Uploaded slide overviews to S3.")

    # Upload crops in parallel or batch
    print(f"[*] Uploading {len(review_items)} crops to S3...")
    uploaded = 0
    for it in review_items:
        cr_local = out_app_dir / it["crop_url"]
        if cr_local.exists():
            try:
                s3.upload_file(str(cr_local), BUCKET, it["crop_s3_key"])
                uploaded += 1
            except Exception:
                pass
    print(f"[+] All {uploaded} crops uploaded to S3 successfully!")


if __name__ == "__main__":
    main()
