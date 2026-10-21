#!/usr/bin/env python3
"""
crop_for_labeling_s3.py
-----------------------
K8s cluster version — reads TIF images directly from S3,
runs YOLO hard-example mining, and uploads 640×640 JPEG crops back to S3.

Memory-efficient: pools store only (key, x, y) metadata.
Images are re-fetched from S3 only at upload time — one at a time.

All config via environment variables (set in the Job manifest):
  S3_ENDPOINT, S3_BUCKET, SOURCE_PREFIX, S3_MODEL_KEY, MODEL_PATH,
  S3_DEST_PREFIX, N_CROPS, HARD_FRAC, CONF_THRESH
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  (from s3-creds secret)
"""

import os
import random
from pathlib import Path
from itertools import groupby

import boto3
import cv2
import numpy as np
from botocore.client import Config

# ── Config from env ────────────────────────────────────────────────────────
S3_ENDPOINT    = os.environ.get("S3_ENDPOINT",    "https://s3.cl4.du.cesnet.cz")
S3_BUCKET      = os.environ.get("S3_BUCKET",      "peg-bucket")
SOURCE_PREFIX  = os.environ.get("SOURCE_PREFIX",
                   "Ostatni/Stomata/extracted/stomata_trait_comparison/new population/")
S3_MODEL_KEY   = os.environ.get("S3_MODEL_KEY",   "")
MODEL_PATH     = Path(os.environ.get("MODEL_PATH", "/tmp/stomata_best.pt"))
S3_DEST_PREFIX = os.environ.get("S3_DEST_PREFIX", "Ostatni/Stomata/crops_for_labeling/")
N_CROPS        = int(os.environ.get("N_CROPS",    "500"))
HARD_FRAC      = float(os.environ.get("HARD_FRAC", "0.60"))
CONF_THRESH    = float(os.environ.get("CONF_THRESH", "0.55"))

CROP_SIZE    = 640
STRIDE       = 400
EDGE_MARGIN  = 60
MIN_VARIANCE = 80
JPEG_QUALITY = 95
SEED         = 42
# ───────────────────────────────────────────────────────────────────────────


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def list_source_keys(s3) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=SOURCE_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".tif"):
                keys.append(obj["Key"])
    return keys


def fetch_image(s3, key: str) -> np.ndarray | None:
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        arr  = np.frombuffer(resp["Body"].read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  ERROR fetching {key}: {e}")
        return None


def build_tiles(w: int, h: int) -> list[tuple[int, int]]:
    xs = list(range(0, w - CROP_SIZE + 1, STRIDE))
    ys = list(range(0, h - CROP_SIZE + 1, STRIDE))
    if xs and xs[-1] + CROP_SIZE < w: xs.append(w - CROP_SIZE)
    if ys and ys[-1] + CROP_SIZE < h: ys.append(h - CROP_SIZE)
    return [(x, y) for x in xs for y in ys]


def tile_variance(img: np.ndarray, x: int, y: int) -> float:
    return float(cv2.cvtColor(
        img[y:y + CROP_SIZE, x:x + CROP_SIZE], cv2.COLOR_BGR2GRAY).var())


def classify_tiles(model, img: np.ndarray,
                   candidates: list[tuple[int, int]]) -> tuple[list, list]:
    results = model(img, verbose=False, conf=0.25)[0]
    detections = []
    if results.masks is not None:
        for mask_xy, box in zip(results.masks.xy, results.boxes):
            pts = np.array(mask_xy)
            detections.append((pts[:, 0].mean(), pts[:, 1].mean(), float(box.conf[0])))
    elif len(results.boxes):
        for box in results.boxes:
            bx = box.xyxy[0].cpu().numpy()
            detections.append(((bx[0]+bx[2])/2, (bx[1]+bx[3])/2, float(box.conf[0])))

    hard, easy = [], []
    for x, y in candidates:
        x2, y2 = x + CROP_SIZE, y + CROP_SIZE
        if tile_variance(img, x, y) < MIN_VARIANCE:
            continue
        tile_dets = [(cx, cy, c) for cx, cy, c in detections
                     if x <= cx <= x2 and y <= cy <= y2]
        # Reject if any stomata centroid bisected at edge
        if any(cx-x < EDGE_MARGIN or x2-cx < EDGE_MARGIN or
               cy-y < EDGE_MARGIN or y2-cy < EDGE_MARGIN
               for cx, cy, _ in tile_dets):
            continue
        if not tile_dets:
            hard.append((x, y))
        elif sum(c for _, _, c in tile_dets) / len(tile_dets) < CONF_THRESH:
            hard.append((x, y))
        else:
            easy.append((x, y))
    return hard, easy


def main():
    random.seed(SEED)
    s3 = make_s3()

    # ── Fetch model ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists() and S3_MODEL_KEY:
        print(f"Downloading model s3://{S3_BUCKET}/{S3_MODEL_KEY} ...")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(MODEL_PATH))
        print("  Model ready.")

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    print("YOLO model loaded — hard-example mining active.\n")

    # ── List source images ─────────────────────────────────────────────────
    print("Listing source images on S3 ...")
    keys = list_source_keys(s3)
    print(f"Found {len(keys)} TIF files.\n")
    random.shuffle(keys)

    n_hard = int(N_CROPS * HARD_FRAC)
    n_easy = N_CROPS - n_hard
    print(f"Target: {N_CROPS} crops  ({n_hard} hard  +  {n_easy} easy)\n")

    # Pools store ONLY metadata — no images kept in RAM
    hard_pool: list[tuple[str, int, int]] = []   # (s3_key, x, y)
    easy_pool: list[tuple[str, int, int]] = []
    scanned = 0

    # ── Phase 1: Scan — build metadata pools ──────────────────────────────
    for key in keys:
        # Stop once we have 2× enough in each pool
        if len(hard_pool) >= n_hard * 2 and len(easy_pool) >= n_easy * 2:
            print(f"Pools full after {scanned} images — stopping scan.")
            break

        img = fetch_image(s3, key)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w < CROP_SIZE or h < CROP_SIZE:
            continue

        candidates = build_tiles(w, h)
        random.shuffle(candidates)
        hard, easy = classify_tiles(model, img, candidates)
        del img   # ← free memory immediately after inference

        for x, y in hard:
            hard_pool.append((key, x, y))
        for x, y in easy:
            easy_pool.append((key, x, y))

        scanned += 1
        if scanned % 20 == 0:
            print(f"  Scanned {scanned}  |  hard: {len(hard_pool)}  "
                  f"easy: {len(easy_pool)}")

    print(f"\nPool sizes — hard: {len(hard_pool)}  easy: {len(easy_pool)}")

    # ── Phase 2: Sample ────────────────────────────────────────────────────
    random.shuffle(hard_pool)
    random.shuffle(easy_pool)
    selected = hard_pool[:n_hard] + easy_pool[:n_easy]
    print(f"Selected {len(selected)} crops to upload.\n")

    # ── Phase 3: Re-fetch & upload (grouped by source key to minimise S3 GETs)
    print(f"Uploading to s3://{S3_BUCKET}/{S3_DEST_PREFIX} ...")
    selected.sort(key=lambda t: t[0])   # group by key
    uploaded = 0

    for key, tiles_iter in groupby(selected, key=lambda t: t[0]):
        tiles = list(tiles_iter)
        img = fetch_image(s3, key)
        if img is None:
            continue
        stem = Path(key).stem.replace(" ", "_")
        for _, x, y in tiles:
            crop = img[y:y + CROP_SIZE, x:x + CROP_SIZE]
            ok, buf = cv2.imencode(".jpg", crop,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            s3_key = f"{S3_DEST_PREFIX}{stem}_x{x}_y{y}.jpg"
            s3.put_object(Bucket=S3_BUCKET, Key=s3_key,
                          Body=buf.tobytes(), ContentType="image/jpeg")
            uploaded += 1
            if uploaded % 50 == 0:
                print(f"  [{uploaded}/{len(selected)}] uploaded")
        del img   # free after all crops from this image are done

    print(f"\n✓ Done — {uploaded} crops at s3://{S3_BUCKET}/{S3_DEST_PREFIX}")


if __name__ == "__main__":
    main()
