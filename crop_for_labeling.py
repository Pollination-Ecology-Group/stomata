#!/usr/bin/env python3
"""
crop_for_labeling.py  –  v2
----------------------------
Generate 640×640 JPEG crops from new microscopy TIF images for Roboflow
labeling, prioritising tiles where the YOLO model struggles.

Hard-example strategy
---------------------
For each source image we run YOLO inference once.  A tile is scored as a
"hard" example when:
  • Average detection confidence is LOW  (model is uncertain), OR
  • Detections are present but all confidence < HARD_CONF_THRESH, OR
  • No detections at all in a region that visually looks like tissue
    (non-trivial pixel variance — avoids blank/background tiles).

Mix:  HARD_FRAC (default 60%) of crops come from hard tiles,
      the rest are random-grid crops for diversity.

Stomata-aware cropping
----------------------
Tiles are rejected if a detected stomata centroid falls within EDGE_MARGIN
pixels of the tile border (would be bisected at label time).

S3 upload
---------
After generation the crops are uploaded to
  s3://{S3_BUCKET}/{S3_PREFIX}crops_for_labeling/
using the same endpoint / credentials as run_inference_s3.py.

Usage
-----
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  python crop_for_labeling.py [--n 500] [--out-dir /tmp/crops] [--no-upload]

Required env vars for S3 (same as run_inference_s3.py):
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  S3_ENDPOINT   (default: https://s3.cl4.du.cesnet.cz)
  S3_BUCKET     (default: bucket)
  S3_MODEL_KEY  – S3 key to best.pt  (used if local model not found)
  MODEL_PATH    (default: /tmp/best.pt)
"""

import argparse
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import boto3
import cv2
import numpy as np
from botocore.client import Config

# ── Config ─────────────────────────────────────────────────────────────────
SOURCE_DIR      = Path("/home/meow/cesnet_data/Ostatni/Stomata/extracted/"
                       "stomata_trait_comparison/new population")
CROP_SIZE       = 640
STRIDE          = 400       # tile overlap = 640-400 = 240 px
EDGE_MARGIN     = 60        # px — reject tile that bisects a stomata centroid
TOTAL_CROPS     = 500
HARD_FRAC       = 0.60      # 60 % of crops come from hard/uncertain tiles
HARD_CONF_THRESH = 0.55     # confidence below this → hard tile
MIN_VARIANCE    = 80        # ignore near-blank tiles (background only)
JPEG_QUALITY    = 95
SEED            = 42

S3_ENDPOINT  = os.environ.get("S3_ENDPOINT",  "https://s3.cl4.du.cesnet.cz")
S3_BUCKET    = os.environ.get("S3_BUCKET",    "bucket")
S3_PREFIX    = os.environ.get("TILE_PREFIX",  "Ostatni/Stomata/")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "")
MODEL_PATH   = Path(os.environ.get("MODEL_PATH", "/tmp/best.pt"))
S3_DEST_PREFIX = f"{S3_PREFIX}crops_for_labeling/"
# ───────────────────────────────────────────────────────────────────────────


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


def fetch_model(s3) -> Path | None:
    """Return path to local model, downloading from S3 if needed."""
    if MODEL_PATH.exists():
        print(f"  Using local model: {MODEL_PATH}")
        return MODEL_PATH
    if S3_MODEL_KEY:
        print(f"  Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(MODEL_PATH))
        print(f"  Model saved to {MODEL_PATH}")
        return MODEL_PATH
    print("  No model found locally or on S3 — falling back to grid-only mode.")
    return None


def build_tile_candidates(img_w: int, img_h: int) -> list[tuple[int, int]]:
    xs = list(range(0, img_w - CROP_SIZE + 1, STRIDE))
    ys = list(range(0, img_h - CROP_SIZE + 1, STRIDE))
    if xs and xs[-1] + CROP_SIZE < img_w:
        xs.append(img_w - CROP_SIZE)
    if ys and ys[-1] + CROP_SIZE < img_h:
        ys.append(img_h - CROP_SIZE)
    return [(x, y) for x in xs for y in ys]


def tile_variance(img_bgr: np.ndarray, x: int, y: int) -> float:
    crop = img_bgr[y:y + CROP_SIZE, x:x + CROP_SIZE]
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).var())


def classify_tiles(model, img_bgr: np.ndarray,
                   candidates: list[tuple[int, int]]
                   ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Return (hard_tiles, easy_tiles) after running inference.
    A tile is 'hard' when the model is uncertain or misses detections.
    Tiles that bisect a stomata centroid are dropped from both lists.
    """
    results = model(img_bgr, verbose=False, conf=0.25)[0]

    # Collect detections: (cx, cy, conf)
    detections: list[tuple[float, float, float]] = []
    if results.masks is not None:
        for mask_xy, box in zip(results.masks.xy, results.boxes):
            pts = np.array(mask_xy)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            detections.append((float(cx), float(cy), float(box.conf[0])))
    elif len(results.boxes):
        for box in results.boxes:
            bx = box.xyxy[0].cpu().numpy()
            cx = (bx[0] + bx[2]) / 2
            cy = (bx[1] + bx[3]) / 2
            detections.append((float(cx), float(cy), float(box.conf[0])))

    hard, easy = [], []
    for x, y in candidates:
        x2, y2 = x + CROP_SIZE, y + CROP_SIZE

        # Variance guard — skip blank tiles
        if tile_variance(img_bgr, x, y) < MIN_VARIANCE:
            continue

        # Stomata-bisection guard
        bisects = False
        tile_dets = [(cx, cy, c) for cx, cy, c in detections
                     if x <= cx <= x2 and y <= cy <= y2]
        for cx, cy, _ in tile_dets:
            if (cx - x < EDGE_MARGIN or x2 - cx < EDGE_MARGIN or
                    cy - y < EDGE_MARGIN or y2 - cy < EDGE_MARGIN):
                bisects = True
                break
        if bisects:
            continue

        # Hard-example scoring
        if not tile_dets:
            # No detections — model missed everything in this tile
            hard.append((x, y))
        else:
            avg_conf = sum(c for _, _, c in tile_dets) / len(tile_dets)
            if avg_conf < HARD_CONF_THRESH:
                hard.append((x, y))
            else:
                easy.append((x, y))

    return hard, easy


def classify_tiles_no_model(img_bgr: np.ndarray,
                             candidates: list[tuple[int, int]]
                             ) -> list[tuple[int, int]]:
    """Without a model, just filter out blank tiles."""
    return [(x, y) for x, y in candidates
            if tile_variance(img_bgr, x, y) >= MIN_VARIANCE]


def upload_to_s3(s3, local_dir: Path) -> int:
    jpgs = list(local_dir.glob("*.jpg"))
    print(f"\nUploading {len(jpgs)} files to "
          f"s3://{S3_BUCKET}/{S3_DEST_PREFIX} ...")
    for i, jpg in enumerate(jpgs, 1):
        key = S3_DEST_PREFIX + jpg.name
        s3.upload_file(str(jpg), S3_BUCKET, key)
        if i % 50 == 0 or i == len(jpgs):
            print(f"  [{i}/{len(jpgs)}] uploaded")
    return len(jpgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=TOTAL_CROPS)
    ap.add_argument("--out-dir", default="/tmp/crops_for_labeling")
    ap.add_argument("--no-upload", action="store_true",
                    help="Skip S3 upload (just save locally)")
    ap.add_argument("--hard-frac", type=float, default=HARD_FRAC,
                    help="Fraction of crops from hard/uncertain tiles (0–1)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    # ── S3 client & model ──────────────────────────────────────────────────
    s3 = s3_client()
    model = None
    local_model = fetch_model(s3)
    if local_model:
        try:
            from ultralytics import YOLO
            model = YOLO(str(local_model))
            print("  YOLO model loaded — hard-example mining enabled.\n")
        except ImportError:
            print("  WARNING: ultralytics not installed; grid-only mode.\n")

    # ── Source images ──────────────────────────────────────────────────────
    all_tifs = sorted(SOURCE_DIR.glob("*.tif"))
    if not all_tifs:
        sys.exit(f"No .tif files found in {SOURCE_DIR}")
    print(f"Found {len(all_tifs)} source images.")
    random.shuffle(all_tifs)

    n_hard = int(args.n * args.hard_frac)
    n_easy = args.n - n_hard
    print(f"Target: {args.n} crops  "
          f"({n_hard} hard / uncertain  +  {n_easy} random)\n")

    hard_pool: list[tuple[Path, int, int]] = []   # (path, x, y)
    easy_pool: list[tuple[Path, int, int]] = []

    # ── Build pools ────────────────────────────────────────────────────────
    print("Scanning images ...")
    for i, tif_path in enumerate(all_tifs, 1):
        img_bgr = cv2.imread(str(tif_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        if w < CROP_SIZE or h < CROP_SIZE:
            continue

        candidates = build_tile_candidates(w, h)

        if model:
            hard, easy = classify_tiles(model, img_bgr, candidates)
            for x, y in hard:
                hard_pool.append((tif_path, x, y))
            for x, y in easy:
                easy_pool.append((tif_path, x, y))
        else:
            good = classify_tiles_no_model(img_bgr, candidates)
            for x, y in good:
                easy_pool.append((tif_path, x, y))

        if i % 50 == 0:
            print(f"  Scanned {i}/{len(all_tifs)}  |  "
                  f"hard pool: {len(hard_pool)}  easy pool: {len(easy_pool)}")

        # Early exit once we have enough in both pools
        if len(hard_pool) >= n_hard * 3 and len(easy_pool) >= n_easy * 3:
            print(f"  Pools large enough after {i} images — stopping scan.")
            break

    print(f"\nPool sizes — hard: {len(hard_pool)}  easy: {len(easy_pool)}")

    # ── Sample ─────────────────────────────────────────────────────────────
    random.shuffle(hard_pool)
    random.shuffle(easy_pool)

    selected = hard_pool[:n_hard] + easy_pool[:n_easy]
    if len(selected) < args.n:
        print(f"WARNING: only {len(selected)} crops available (requested {args.n})")

    # ── Save crops ─────────────────────────────────────────────────────────
    print(f"\nSaving {len(selected)} crops to {out_dir} ...")
    saved = 0
    for tif_path, x, y in selected:
        img_bgr = cv2.imread(str(tif_path))
        if img_bgr is None:
            continue
        crop = img_bgr[y:y + CROP_SIZE, x:x + CROP_SIZE]
        stem = tif_path.stem.replace(" ", "_")
        out_name = out_dir / f"{stem}_x{x}_y{y}.jpg"
        cv2.imwrite(str(out_name), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1
        if saved % 100 == 0:
            print(f"  Saved {saved}/{len(selected)}")

    print(f"\n{saved} crops saved.")

    # ── S3 upload ──────────────────────────────────────────────────────────
    if not args.no_upload:
        uploaded = upload_to_s3(s3, out_dir)
        print(f"\n✓ {uploaded} crops at "
              f"s3://{S3_BUCKET}/{S3_DEST_PREFIX}")
    else:
        print(f"\nSkipped upload. Files in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
