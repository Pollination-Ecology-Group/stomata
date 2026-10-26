#!/usr/bin/env python3
"""
render_clean_overviews.py
Generates high-fidelity annotated panoramic overviews for the 10 review slides
using native-resolution sliding-window tiled inference with stomata_best_latest.pt.
Draws crisp vector contours and labels with NO solid orange fills.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.append(str(SRC_DIR))
from extract_traits import predict_tiled, PX_TO_UM

RAW_DIR = Path("/home/meow/cesnet_new/Ostatni/Stomata/detect/stomata_trait_comparison_AA012-20251216T171633Z-3-002/stomata_trait_comparison_AA012")
MODEL_PATH = Path("/tmp/stomata_best_latest.pt")
OUT_DIR = Path(__file__).resolve().parent / "src" / "review_app" / "static" / "overviews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_STEMS = [
    "2-4-C_AD_3", "5-10-D_AB_3", "5-10-D_AB_4", "5-10-D_AD_3",
    "5-9-D_AB_3", "5-9-E_AB_1", "5-9-E_AD_2", "6-3-C_AB_3",
    "6-3-D_AB_2", "6-4-A_AD_1"
]

def render_slide(stem, model):
    matches = list(RAW_DIR.glob(f"{stem}.*"))
    if not matches:
        print(f"[-] Could not find raw image for {stem}")
        return None
    
    img_path = matches[0]
    t0 = time.time()
    print(f"[*] Processing {img_path.name} ({img_path.stat().st_size / 1e6:.1f} MB)...", flush=True)
    
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[-] Failed to load {img_path}")
        return None
    
    H, W = img_bgr.shape[:2]
    
    # Run native tiled inference
    stomas, centers = predict_tiled(
        model, img_bgr,
        tile_size=640, stride=448,
        conf_stoma=0.18, conf_center=0.38,
        enhance_contrast=True
    )
    
    dt = time.time() - t0
    print(f"    -> Found {len(stomas)} stomata and {len(centers)} centers in {dt:.1f}s", flush=True)
    
    vis = img_bgr.copy()
    overlay = img_bgr.copy()
    
    # Match stomas to pores
    matched_pairs = []
    for s_idx, (s_poly, s_conf) in enumerate(stomas, 1):
        matched_c = None
        for c_poly, c_conf in centers:
            if s_poly.contains(c_poly.centroid) or (s_poly.intersection(c_poly).area / max(1.0, c_poly.area) > 0.5):
                matched_c = (c_poly, c_conf)
                break
        matched_pairs.append((s_poly, s_conf, matched_c))
    
    # Draw visual annotations
    for idx, (s_poly, s_conf, c_tuple) in enumerate(matched_pairs, 1):
        s_pts = np.array(s_poly.exterior.coords, dtype=np.int32)
        
        # 1. Subtle stoma fill on overlay (soft translucent emerald)
        cv2.fillPoly(overlay, [s_pts], (34, 197, 94))
        
        # 2. Crisp boundary contour (Green/Emerald)
        cv2.polylines(vis, [s_pts], isClosed=True, color=(34, 197, 94), thickness=2, lineType=cv2.LINE_AA)
        
        # 3. Draw pore contour if matched (Cyan)
        if c_tuple is not None:
            c_poly, c_conf = c_tuple
            c_pts = np.array(c_poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [c_pts], (248, 189, 56))  # subtle cyan/blue in BGR
            cv2.polylines(vis, [c_pts], isClosed=True, color=(248, 189, 56), thickness=2, lineType=cv2.LINE_AA)
        
        # 4. Centroid marker and badge
        scx, scy = int(s_poly.centroid.x), int(s_poly.centroid.y)
        cv2.circle(vis, (scx, scy), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        
        # Calculate length in um
        rect = cv2.minAreaRect(np.array(s_poly.exterior.coords, dtype=np.float32))
        maj_um = max(rect[1][0], rect[1][1]) * PX_TO_UM
        
        label = f"#{idx} ({maj_um:.1f}um)"
        font_scale = 0.55
        thickness = 1
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Label pill background
        lx, ly = max(10, scx - lw // 2), max(lh + 6, scy - 18)
        cv2.rectangle(overlay, (lx - 3, ly - lh - 3), (lx + lw + 3, ly + 3), (15, 23, 42), -1)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Blend overlay with 0.15 opacity for soft professional glow, no opaque orange blobs!
    alpha = 0.18
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    
    # Save output
    out_file = OUT_DIR / f"{stem}_overview.jpg"
    cv2.imwrite(str(out_file), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[+] Saved clean overview to {out_file.name}", flush=True)
    return len(stomas)

def main():
    print(f"[*] Loading model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))
    
    total_stomata = 0
    for stem in SAMPLE_STEMS:
        n = render_slide(stem, model)
        if n:
            total_stomata += n
            
    print(f"\n[DONE] Rendered all clean overviews with {total_stomata} total stomata detected across 10 micrographs!")

if __name__ == "__main__":
    main()
