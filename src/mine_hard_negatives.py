#!/usr/bin/env python3
"""
mine_hard_negatives.py
Active Learning & Hard Negative Mining tool for stomata detection.

Scans unannotated microscopy images to automatically discover and extract:
1. Hard Negative Backgrounds: High-texture pavement cell patches where NO stomata are present.
2. Hard Positive Crops: Stomata with borderline confidence or ambiguous pore geometry.
3. Staged crops formatted for direct upload to Roboflow or S3 backgrounds/.
"""

import os
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

from extract_traits import predict_tiled


def compute_texture_energy(crop_bgr):
    """Computes Laplacian variance (focus/texture score) of a patch."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def mine_hard_samples(model_path, input_dir, output_dir, max_samples_per_image=5, tile_size=640):
    output_path = Path(output_dir)
    bg_out = output_path / "hard_negatives_background"
    borderline_out = output_path / "borderline_stomata"
    bg_out.mkdir(parents=True, exist_ok=True)
    borderline_out.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model: {model_path}")
    model = YOLO(model_path)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True) +
                         glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True))

    print(f"[*] Scanning {len(image_paths)} images for active learning candidates...")

    total_bg_mined = 0
    total_borderline_mined = 0

    for img_idx, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_bgr.shape[:2]

        # 1. Run inference
        stomas, centers = predict_tiled(model, img_rgb, tile_size=tile_size, conf=0.15)

        # 2. Identify borderline stomata (low confidence or multiple centers)
        borderline_count = 0
        for s_idx, (stoma_poly, conf) in enumerate(stomas):
            if borderline_count >= max_samples_per_image:
                break

            centers_inside = [c for c, _ in centers if stoma_poly.contains(c.centroid)]
            is_borderline = (0.20 <= conf <= 0.60) or (len(centers_inside) > 1) or (len(centers_inside) == 0 and conf > 0.5)

            if is_borderline:
                cx, cy = int(stoma_poly.centroid.x), int(stoma_poly.centroid.y)
                half = tile_size // 2
                x1 = max(0, min(W - tile_size, cx - half))
                y1 = max(0, min(H - tile_size, cy - half))
                crop = img_bgr[y1:y1+tile_size, x1:x1+tile_size]

                out_name = f"{stem}_stoma_hard_{s_idx}_conf{int(conf*100)}.png"
                cv2.imwrite(str(borderline_out / out_name), crop)
                borderline_count += 1
                total_borderline_mined += 1

        # 3. Mine hard negative backgrounds (dense pavement cell walls, but zero stomata)
        bg_count = 0
        # Sample candidate grid points across image
        for y in range(0, H - tile_size, tile_size):
            for x in range(0, W - tile_size, tile_size):
                if bg_count >= max_samples_per_image:
                    break

                tile_poly = Polygon([(x, y), (x + tile_size, y), (x + tile_size, y + tile_size), (x, y + tile_size)])

                # Check if any stoma is inside or touches this tile
                has_stoma = any(tile_poly.intersects(stoma_poly) for stoma_poly, _ in stomas)

                if not has_stoma:
                    crop = img_bgr[y:y+tile_size, x:x+tile_size]
                    texture = compute_texture_energy(crop)
                    # Only keep high-texture tiles (avoid blank margins or out-of-focus empty patches)
                    if texture > 80.0:
                        out_name = f"{stem}_bg_neg_{x}_{y}.png"
                        cv2.imwrite(str(bg_out / out_name), crop)
                        bg_count += 1
                        total_bg_mined += 1

        print(f"[{img_idx+1}/{len(image_paths)}] {stem}: mined {bg_count} hard backgrounds, {borderline_count} borderline stomata.")

    print("\n" + "=" * 60)
    print(f"[+] Mining Complete!")
    print(f"    Hard negative background crops: {total_bg_mined} -> {bg_out}")
    print(f"    Borderline stomata crops:       {total_borderline_mined} -> {borderline_out}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives and active learning candidates.")
    parser.add_argument("--model", type=str, required=True, help="Path to current YOLO segmentation model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of unannotated images")
    parser.add_argument("--output_dir", type=str, default="./mined_samples", help="Output directory")
    args = parser.parse_args()

    mine_hard_samples(args.model, args.input_dir, args.output_dir)
