#!/usr/bin/env python3
"""
train_stomata_s3.py
Automated training pipeline for Stomata instance segmentation on CESNET Kubernetes.

Workflow:
1. Connects to CESNET S3 storage.
2. Downloads the staged Roboflow dataset zip from S3 (e.g. Stomata.v5i.yolov8.zip).
3. Downloads empty background images from S3 (e.g. Ostatni/Stomata/backgrounds/).
4. Unpacks dataset, merges background images with empty labels, and creates a
   stratified train/validation split (default 85% train / 15% val).
5. Generates the dataset YAML with classes {0: 'center', 1: 'stoma'}.
6. Trains YOLO segmentation model (YOLOv8-seg or YOLO11-seg) with microscopy augmentations.
7. Evaluates the best model and uploads the new weights and training metrics/plots back to S3.
"""

import os
import sys
import glob
import shutil
import zipfile
import random
import yaml
import smtplib
from email.message import EmailMessage
from datetime import datetime
from pathlib import Path
import boto3
from botocore.client import Config
from ultralytics import YOLO, settings

# ---------------------------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------------------------
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.getenv("S3_BUCKET", "peg-bucket")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_STAGED_PREFIX = os.getenv("S3_STAGED_PREFIX", "Ostatni/Stomata/staged_area/")
S3_BACKGROUNDS_PREFIX = os.getenv("S3_BACKGROUNDS_PREFIX", "Ostatni/Stomata/backgrounds/")
S3_MODELS_PREFIX = os.getenv("S3_MODELS_PREFIX", "Ostatni/Stomata/trained_models/")
S3_RUNS_PREFIX = os.getenv("S3_RUNS_PREFIX", "Ostatni/Stomata/training_runs/")

BASE_MODEL = os.getenv("BASE_MODEL", "yolo11m-seg.pt")
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
EPOCHS = int(os.getenv("EPOCHS", "120"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
PATIENCE = int(os.getenv("PATIENCE", "30"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.15"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/stomata_train"))

# Ensure working directory exists and configure Ultralytics settings to write within writable /tmp
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_DIR)
settings.update({
    "weights_dir": str(WORK_DIR / "weights"),
    "runs_dir": str(WORK_DIR / "runs"),
    "datasets_dir": str(WORK_DIR / "datasets"),
})


def get_s3_client():
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("[-] Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set!")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def send_email_notification(subject, body):
    recipient = os.environ.get("EMAIL_RECIPIENT", "jakubstenc@gmail.com")
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "") if os.environ.get("GMAIL_APP_PASSWORD") else ""
    if not pw:
        print("[!] GMAIL_APP_PASSWORD not set. Skipping email dispatch.")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = recipient
    msg['To'] = recipient

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15)
        server.login(recipient, pw)
        server.send_message(msg)
        server.quit()
        print(f"[+] Email notification sent successfully to {recipient}")
        return True
    except Exception as e:
        print(f"[-] Failed to send email: {e}")
        return False


def find_latest_staged_zip(s3_client):
    print(f"[*] Checking staged area: s3://{S3_BUCKET}/{S3_STAGED_PREFIX}")
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_STAGED_PREFIX)

    zip_objects = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".zip"):
                zip_objects.append((key, obj["LastModified"], obj["Size"]))

    if not zip_objects:
        raise RuntimeError(f"No .zip files found in s3://{S3_BUCKET}/{S3_STAGED_PREFIX}")

    # Sort by LastModified descending
    zip_objects.sort(key=lambda x: x[1], reverse=True)
    latest_key, last_mod, size = zip_objects[0]
    print(f"[+] Found dataset zip: {latest_key} ({size / (1024*1024):.2f} MB, modified {last_mod})")
    return latest_key


def download_backgrounds(s3_client, download_dir: Path):
    print(f"[*] Downloading background images from s3://{S3_BUCKET}/{S3_BACKGROUNDS_PREFIX}...")
    download_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_BACKGROUNDS_PREFIX)

    bg_files = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".png", ".jpg", ".jpeg")):
                filename = os.path.basename(key)
                local_path = download_dir / filename
                s3_client.download_file(S3_BUCKET, key, str(local_path))
                bg_files.append(local_path)

    print(f"[+] Downloaded {len(bg_files)} background images.")
    return bg_files


def prepare_dataset(zip_local_path: Path, bg_files: list, dataset_root: Path):
    print("\n" + "=" * 60)
    print("[*] Preparing YOLO segmentation dataset with background negatives...")
    print("=" * 60)

    temp_extract = WORK_DIR / "temp_extract"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)

    print(f"[*] Extracting {zip_local_path.name}...")
    with zipfile.ZipFile(zip_local_path, "r") as z:
        z.extractall(temp_extract)

    # Search for all image files with corresponding labels in the extracted zip
    labeled_candidates = []
    for root, _, files in os.walk(temp_extract):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = Path(root) / f
                stem = img_path.stem

                # Check potential label locations
                lbl_path = Path(root) / f"{stem}.txt"
                if not lbl_path.exists():
                    lbl_path = Path(root).parent / "labels" / f"{stem}.txt"
                if not lbl_path.exists():
                    lbl_path = Path(root).parent.parent / "labels" / f"{stem}.txt"

                if lbl_path.exists() and lbl_path.stat().st_size > 0:
                    labeled_candidates.append((img_path, lbl_path))

    print(f"[+] Found {len(labeled_candidates)} labeled images in the archive.")

    # Prepare background candidates (label = CREATE_EMPTY)
    bg_candidates = [(bg_path, "CREATE_EMPTY") for bg_path in bg_files]
    print(f"[+] Added {len(bg_candidates)} empty background images.")

    # Set random seed for reproducible stratified split
    random.seed(RANDOM_SEED)
    random.shuffle(labeled_candidates)
    random.shuffle(bg_candidates)

    # Stratified split: labeled
    n_labeled_val = int(len(labeled_candidates) * VAL_SPLIT)
    val_labeled = labeled_candidates[:n_labeled_val]
    train_labeled = labeled_candidates[n_labeled_val:]

    # Stratified split: backgrounds
    n_bg_val = int(len(bg_candidates) * VAL_SPLIT)
    val_bg = bg_candidates[:n_bg_val]
    train_bg = bg_candidates[n_bg_val:]

    train_all = train_labeled + train_bg
    val_all = val_labeled + val_bg

    random.shuffle(train_all)
    random.shuffle(val_all)

    # Destination directories
    splits = {
        "train": dataset_root / "train",
        "valid": dataset_root / "valid",
    }
    for s_path in splits.values():
        (s_path / "images").mkdir(parents=True, exist_ok=True)
        (s_path / "labels").mkdir(parents=True, exist_ok=True)

    def write_split(items, split_name):
        dest_img_dir = splits[split_name] / "images"
        dest_lbl_dir = splits[split_name] / "labels"

        for img_src, lbl_src in items:
            dest_img = dest_img_dir / img_src.name
            shutil.copy2(img_src, dest_img)

            dest_lbl = dest_lbl_dir / f"{img_src.stem}.txt"
            if lbl_src == "CREATE_EMPTY":
                # Create empty file for background images
                dest_lbl.touch()
            else:
                shutil.copy2(lbl_src, dest_lbl)

    write_split(train_all, "train")
    write_split(val_all, "valid")

    # Generate data.yaml with classes: 0 -> center (pore), 1 -> stoma
    yaml_content = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": {
            0: "center",
            1: "stoma",
        },
    }

    yaml_file = dataset_root / "data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("\n[+] Dataset preparation complete:")
    print(f"    Train: {len(train_all)} total ({len(train_labeled)} labeled, {len(train_bg)} backgrounds)")
    print(f"    Valid: {len(val_all)} total ({len(val_labeled)} labeled, {len(val_bg)} backgrounds)")
    print(f"    Config: {yaml_file}")

    # Clean up temp extraction
    shutil.rmtree(temp_extract, ignore_errors=True)
    return yaml_file


def train(yaml_file: Path):
    print("\n" + "=" * 60)
    print(f"[*] Starting YOLO training with base model: {BASE_MODEL}")
    print(f"    Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, ImgSize: {IMG_SIZE}, Patience: {PATIENCE}")
    print("=" * 60)

    # Configure custom Albumentations for out-of-focus microscopy robustness
    try:
        import albumentations as A
        import ultralytics.data.augment as aug

        orig_init = aug.Albumentations.__init__

        def custom_albumentations_init(self, p=1.0, transforms=None, flip_idx=None):
            microscopy_transforms = [
                A.Blur(blur_limit=(3, 9), p=0.35),
                A.MedianBlur(blur_limit=(3, 7), p=0.25),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.40),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.30),
                A.RandomGamma(gamma_limit=(80, 120), p=0.25),
            ]
            orig_init(self, p=p, transforms=microscopy_transforms, flip_idx=flip_idx)

        aug.Albumentations.__init__ = custom_albumentations_init
        print("[+] Injected microscopy Defocus/Blur & CLAHE augmentations into training pipeline.")
    except Exception as e:
        print(f"[-] Could not customize Albumentations: {e}")

    model = YOLO(BASE_MODEL)
    run_dir = WORK_DIR / "runs"
    results = model.train(
        data=str(yaml_file),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=str(run_dir),
        name="stomata_run",
        # Biological augmentations for microscopy images:
        degrees=180.0,       # Rotation invariance: stomata have no canonical orientation
        flipud=0.5,          # Vertical flip
        fliplr=0.5,          # Horizontal flip
        mosaic=0.4,          # Reduced mosaic to preserve fine stomatal pore slits
        close_mosaic=20,     # Disable mosaic for the final 20 epochs to learn crisp boundaries
        scale=0.4,           # Scale variation around native resolution
        box=7.5,             # Bounding box loss gain
        retina_masks=True,   # High-resolution mask generation
        save=True,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise RuntimeError(f"best.pt not found at expected location: {best_weights}")

    print(f"[+] Training complete. Best weights saved at: {best_weights}")
    return best_weights, save_dir


def upload_results(s3_client, best_weights: Path, run_save_dir: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    versioned_model_name = f"stomata_2class_{timestamp}.pt"

    print("\n" + "=" * 60)
    print("[*] Uploading model weights and metrics to CESNET S3...")
    print("=" * 60)

    # 1. Upload timestamped model
    s3_key_versioned = f"{S3_MODELS_PREFIX}{versioned_model_name}"
    print(f"[*] Uploading versioned model: s3://{S3_BUCKET}/{s3_key_versioned}")
    s3_client.upload_file(str(best_weights), S3_BUCKET, s3_key_versioned)

    # 2. Upload latest model pointer
    s3_key_latest = f"{S3_MODELS_PREFIX}stomata_best_latest.pt"
    print(f"[*] Updating latest model pointer: s3://{S3_BUCKET}/{s3_key_latest}")
    s3_client.upload_file(str(best_weights), S3_BUCKET, s3_key_latest)

    # 3. Upload training plots and metrics
    s3_run_prefix = f"{S3_RUNS_PREFIX}stomata_{timestamp}/"
    print(f"[*] Uploading training artifacts to s3://{S3_BUCKET}/{s3_run_prefix}")

    metric_extensions = (".png", ".jpg", ".jpeg", ".csv")
    for file_path in run_save_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in metric_extensions:
            dest_key = f"{s3_run_prefix}{file_path.name}"
            s3_client.upload_file(str(file_path), S3_BUCKET, dest_key)

    print("\n[+] All artifacts successfully uploaded to CESNET S3!")
    print(f"    Versioned model: s3://{S3_BUCKET}/{s3_key_versioned}")
    print(f"    Latest model:    s3://{S3_BUCKET}/{s3_key_latest}")
    print(f"    Run metrics:     s3://{S3_BUCKET}/{s3_run_prefix}")

    # 4. Dispatch Email Notification
    subject = f"✅ Stomata YOLO Retraining Completed: {versioned_model_name} (v7 + Backgrounds)"
    email_body = f"""Hello Jakub,

Your Stomata YOLO retraining run on the CESNET GPU cluster has completed successfully!

🎯 Model Details:
══════════════════════════════════════════════════════════════
• Model Version: {versioned_model_name}
• Base Architecture: {BASE_MODEL}
• Training Epochs: {EPOCHS} (ImgSize: {IMG_SIZE})
• Latest Model Pointer: s3://{S3_BUCKET}/{s3_key_latest}
• Versioned Weights: s3://{S3_BUCKET}/{s3_key_versioned}
• Run Metrics Directory: s3://{S3_BUCKET}/{s3_run_prefix}

Full batch inference across all 3,253 micrographs will now execute using these new weights.

Best regards,
CESNET Automated Retraining Pipeline
"""
    send_email_notification(subject, email_body)


def main():
    print(f"[*] Stomata Retraining Job Initialized")
    print(f"    S3 Endpoint: {S3_ENDPOINT}")
    print(f"    S3 Bucket:   {S3_BUCKET}")
    print(f"    Work Dir:    {WORK_DIR}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()

    # 1. Find and download latest staged dataset zip
    latest_zip_key = find_latest_staged_zip(s3)
    local_zip_path = WORK_DIR / Path(latest_zip_key).name
    print(f"[*] Downloading {latest_zip_key} -> {local_zip_path}...")
    s3.download_file(S3_BUCKET, latest_zip_key, str(local_zip_path))

    # 2. Download background negative images
    bg_download_dir = WORK_DIR / "raw_backgrounds"
    bg_files = download_backgrounds(s3, bg_download_dir)

    # 3. Assemble YOLO dataset with stratified background split
    dataset_dir = WORK_DIR / "dataset"
    yaml_file = prepare_dataset(local_zip_path, bg_files, dataset_dir)

    # 4. Train model
    best_weights, run_save_dir = train(yaml_file)

    # 5. Upload results
    upload_results(s3, best_weights, run_save_dir)

    print("\n" + "=" * 60)
    print("✅ RETRAINING JOB FINISHED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
