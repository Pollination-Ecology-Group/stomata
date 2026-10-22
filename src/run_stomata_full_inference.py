#!/usr/bin/env python3
"""
run_stomata_full_inference.py
Complete batch trait extraction across all 3,253 micrographs on CESNET GPU.
Merges detections directly with experimental metadata (cytotypes, ploidy, populations, leaf side).
Saves periodic checkpoints to S3 and sends an email notification upon completion.
"""

import os
import io
import time
import smtplib
import argparse
import concurrent.futures
from email.message import EmailMessage
from pathlib import Path

import boto3
from botocore.client import Config
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from ultralytics import YOLO

# ── Import trait extraction functions ──────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extract_traits import (
    process_image,
    PX_TO_UM,
    compute_focus_score,
    get_minor_axis_line,
    measure_subpixel_pore_aperture,
    compute_gsmax,
    compute_spi,
    compute_clark_evans
)

# ── Configuration ─────────────────────────────────────────────────────────────
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "Ostatni/Stomata/trained_models/stomata_best_latest.pt")
S3_META_KEY = os.environ.get("S3_META_KEY", "Ostatni/Stomata/metadata/stomata_sampling_trait_comparison_by_sample.xlsx")
MODEL_LOCAL = os.environ.get("MODEL_LOCAL", "/tmp/stomata_best_latest.pt")
META_LOCAL = os.environ.get("META_LOCAL", "/tmp/stomata_metadata.xlsx")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/inference_results")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "Ostatni/Stomata/inference_results")

FETCH_WORKERS = int(os.environ.get("FETCH_WORKERS", "16"))
TILE_SIZE = int(os.environ.get("TILE_SIZE", "640"))
STRIDE = int(os.environ.get("STRIDE", "448"))
CONF_STOMA = float(os.environ.get("CONF_STOMA", "0.18"))
CONF_CENTER = float(os.environ.get("CONF_CENTER", "0.38"))
MIN_FOCUS = float(os.environ.get("MIN_FOCUS", "2.8"))
ENHANCE_CONTRAST = os.environ.get("ENHANCE_CONTRAST", "true").lower() in ("true", "1", "yes")

EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT", "jakubstenc@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
    )


def send_email_notification(subject, body):
    pw = GMAIL_APP_PASSWORD.replace(" ", "") if GMAIL_APP_PASSWORD else ""
    if not pw:
        print("[!] GMAIL_APP_PASSWORD not set. Skipping email dispatch.")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_RECIPIENT
    msg['To'] = EMAIL_RECIPIENT

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15)
        server.login(EMAIL_RECIPIENT, pw)
        server.send_message(msg)
        server.quit()
        print(f"[+] Email notification sent successfully to {EMAIL_RECIPIENT}")
        return True
    except Exception as e:
        print(f"[-] Failed to send email: {e}")
        return False


def download_single_image(s3, s3_key, max_retries=3):
    for attempt in range(max_retries):
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            buf = obj['Body'].read()
            img_array = np.frombuffer(buf, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                return img_bgr
            # Fallback via PIL if OpenCV imdecode fails for complex TIFF tags
            pil_img = Image.open(io.BytesIO(buf)).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[-] Error downloading {s3_key}: {e}", flush=True)
                return None
            time.sleep(1.0)
    return None


def run_full_inference(limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    s3 = get_s3_client()

    # 1. Download Model Checkpoint
    if not os.path.exists(MODEL_LOCAL):
        print(f"[*] Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}...", flush=True)
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_LOCAL)
    print(f"[+] Model ready at {MODEL_LOCAL} ({os.path.getsize(MODEL_LOCAL):,} bytes)", flush=True)
    model = YOLO(MODEL_LOCAL)

    # 2. Download Metadata
    if not os.path.exists(META_LOCAL):
        print(f"[*] Downloading metadata from s3://{S3_BUCKET}/{S3_META_KEY}...", flush=True)
        s3.download_file(S3_BUCKET, S3_META_KEY, META_LOCAL)
    print(f"[+] Metadata loaded from {META_LOCAL}", flush=True)

    meta_df = pd.read_excel(META_LOCAL, sheet_name="stomata_photos")
    if limit is not None and limit > 0:
        meta_df = meta_df.head(limit)
    total_imgs = len(meta_df)
    print(f"[*] Scheduled {total_imgs} images for processing.", flush=True)

    # Fast metadata lookup by image_file
    meta_lookup = meta_df.set_index("image_file").to_dict(orient="index")

    all_stomata_records = []
    summary_records = []

    start_time = time.time()
    processed_count = 0

    # Build S3 keys
    tasks = []
    for _, row in meta_df.iterrows():
        folder_clean = str(row['folder']).replace('\\', '/').strip('/')
        img_file = str(row['image_file'])
        s3_key = f"Ostatni/Stomata/extracted/{folder_clean}/{img_file}"
        tasks.append((img_file, s3_key, row.to_dict()))

    # Process in batches to pipeline downloading & GPU inference
    batch_size = 50
    for b_idx in range(0, total_imgs, batch_size):
        b_tasks = tasks[b_idx: b_idx + batch_size]
        downloaded = {}

        # Parallel download batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            future_to_task = {
                executor.submit(download_single_image, s3, task[1]): task
                for task in b_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                img_bgr = future.result()
                if img_bgr is not None:
                    downloaded[task[0]] = (img_bgr, task[2])

        # Run inference on downloaded batch
        for img_name, (img_bgr, row_meta) in downloaded.items():
            processed_count += 1
            stoma_res = process_image(
                model, img_bgr, img_name,
                tile_size=TILE_SIZE, stride=STRIDE,
                conf_stoma=CONF_STOMA, conf_center=CONF_CENTER,
                enhance_contrast=ENHANCE_CONTRAST,
                min_focus_for_sizing=MIN_FOCUS
            )

            # Inject metadata into per-stoma records
            for r in stoma_res:
                r.update({
                    "plant_ID": row_meta.get("plant_ID"),
                    "population": row_meta.get("population"),
                    "cytotype": row_meta.get("cytotype"),
                    "ploidy": row_meta.get("ploidy"),
                    "leaf_side": row_meta.get("leaf_side"),
                    "side_code": row_meta.get("side_code"),
                    "photo_no": row_meta.get("photo_no"),
                    "cultivation_run": row_meta.get("cultivation_run"),
                    "cultivation_number": row_meta.get("cultivation_number"),
                    "file_qualifier": row_meta.get("file_qualifier")
                })
            all_stomata_records.extend(stoma_res)

            # Compute image summary metrics
            tot_count = sum(1 for r in stoma_res if r["counts_for_density"])
            edge_count = sum(1 for r in stoma_res if r["is_edge_stoma"])
            blur_count = sum(1 for r in stoma_res if not r["is_well_focused"])
            field_area_mm2 = 0.1573
            density = round(tot_count / field_area_mm2, 1)

            ploidy_sub = [r for r in stoma_res if r["is_suitable_for_ploidy_sizing"]]
            n_ploidy = len(ploidy_sub)

            lengths = [r["stoma_major_um"] for r in ploidy_sub]
            widths = [r["stoma_minor_um"] for r in ploidy_sub]
            gc_widths = [r["guard_cell_width_um"] for r in ploidy_sub]
            apertures = [r["pore_minor_um"] for r in ploidy_sub]

            median_len = round(float(np.median(lengths)), 2) if n_ploidy > 0 else np.nan
            median_pore_len = round(float(np.median([r["pore_major_um"] for r in ploidy_sub])), 2) if n_ploidy > 0 else np.nan
            median_aperture = round(float(np.median(apertures)), 2) if n_ploidy > 0 else np.nan
            median_gc_width = round(float(np.median(gc_widths)), 2) if n_ploidy > 0 else np.nan

            gsmax = compute_gsmax(density, median_pore_len, median_aperture, median_gc_width) if n_ploidy > 0 else np.nan
            spi = compute_spi(density, median_len) if (not np.isnan(median_len)) else np.nan

            centroids = [(r["centroid_x"], r["centroid_y"]) for r in stoma_res]
            mean_nnd, ce_r = compute_clark_evans(centroids)

            summary_records.append({
                "image_name": img_name,
                "plant_ID": row_meta.get("plant_ID"),
                "population": row_meta.get("population"),
                "cytotype": row_meta.get("cytotype"),
                "ploidy": row_meta.get("ploidy"),
                "leaf_side": row_meta.get("leaf_side"),
                "side_code": row_meta.get("side_code"),
                "photo_no": row_meta.get("photo_no"),
                "cultivation_run": row_meta.get("cultivation_run"),
                "cultivation_number": row_meta.get("cultivation_number"),
                "file_qualifier": row_meta.get("file_qualifier"),
                "total_stomata_count": tot_count,
                "stomatal_density_per_mm2": density,
                "edge_stomata_count": edge_count,
                "blurred_stomata_count": blur_count,
                "ploidy_sample_count": n_ploidy,
                "ploidy_median_stoma_length_um": median_len,
                "ploidy_mean_stoma_length_um": round(float(np.mean(lengths)), 2) if n_ploidy > 0 else np.nan,
                "ploidy_stoma_length_std_um": round(float(np.std(lengths)), 2) if n_ploidy > 0 else np.nan,
                "ploidy_median_stoma_width_um": round(float(np.median(widths)), 2) if n_ploidy > 0 else np.nan,
                "ploidy_median_guard_cell_width_um": median_gc_width,
                "ploidy_median_aperture_um": median_aperture,
                "gsmax_mol_m2_s": gsmax,
                "stomatal_pore_area_index_spi": spi,
                "mean_nnd_um": mean_nnd,
                "clark_evans_r": ce_r,
            })

        elapsed = time.time() - start_time
        fps = processed_count / max(0.1, elapsed)
        eta_sec = (total_imgs - processed_count) / max(0.01, fps)
        print(f"[{processed_count}/{total_imgs}] {fps:.2f} img/s | Elapsed: {elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m", flush=True)

        # Checkpoint every 100 images
        if processed_count % 100 == 0 or processed_count == total_imgs:
            cp_stoma = os.path.join(OUTPUT_DIR, "checkpoint_traits_per_stoma.csv")
            cp_summ = os.path.join(OUTPUT_DIR, "checkpoint_image_summary.csv")
            pd.DataFrame(all_stomata_records).to_csv(cp_stoma, index=False)
            pd.DataFrame(summary_records).to_csv(cp_summ, index=False)
            try:
                s3.upload_file(cp_stoma, S3_BUCKET, f"{S3_OUTPUT_PREFIX}/checkpoint_traits_per_stoma.csv")
                s3.upload_file(cp_summ, S3_BUCKET, f"{S3_OUTPUT_PREFIX}/checkpoint_image_summary.csv")
                print(f"[+] Checkpoint synced to s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/", flush=True)
            except Exception as e:
                print(f"[-] Checkpoint upload failed: {e}", flush=True)

    # 4. Final Save & S3 Export
    final_stoma_csv = os.path.join(OUTPUT_DIR, "stomata_traits_all_3253_stomata.csv")
    final_summary_csv = os.path.join(OUTPUT_DIR, "stomata_traits_all_3253_image_summary.csv")

    df_stoma = pd.DataFrame(all_stomata_records)
    df_summ = pd.DataFrame(summary_records)

    df_stoma.to_csv(final_stoma_csv, index=False)
    df_summ.to_csv(final_summary_csv, index=False)

    print(f"[+] Final results written locally: {len(df_stoma)} stomata, {len(df_summ)} images", flush=True)

    s3_final_stoma = f"{S3_OUTPUT_PREFIX}/stomata_traits_all_3253_stomata.csv"
    s3_final_summ = f"{S3_OUTPUT_PREFIX}/stomata_traits_all_3253_image_summary.csv"

    s3.upload_file(final_stoma_csv, S3_BUCKET, s3_final_stoma)
    s3.upload_file(final_summary_csv, S3_BUCKET, s3_final_summ)
    print(f"[+] Uploaded final datasets to:\n    s3://{S3_BUCKET}/{s3_final_stoma}\n    s3://{S3_BUCKET}/{s3_final_summ}", flush=True)

    total_time_min = (time.time() - start_time) / 60.0

    # 5. Email Notification
    subject = f"✅ Stomata Inference & Trait Extraction Completed ({processed_count} images)"
    
    # Generate summary stats for email
    cytotype_summary = ""
    try:
        grp = df_summ.groupby(["cytotype", "leaf_side"]).agg({
            "total_stomata_count": "sum",
            "stomatal_density_per_mm2": "mean",
            "ploidy_median_stoma_length_um": "mean"
        }).round(2)
        cytotype_summary = grp.to_string()
    except Exception:
        pass

    email_body = f"""Hello Jakub,

Your automated Stomata trait extraction pipeline on the CESNET GPU cluster has successfully completed!

📊 Execution Summary:
══════════════════════════════════════════════════════════════
• Total Images Processed: {processed_count} / {total_imgs}
• Total Stomata Measured: {len(df_stoma):,}
• High-Precision Ploidy Sizing Stomata: {len(df_stoma[df_stoma['is_suitable_for_ploidy_sizing']]):,}
• Total Wall-Clock Time: {total_time_min:.1f} minutes ({fps:.2f} images/sec)

🔬 Preliminary Cytotype & Leaf Side Comparison:
──────────────────────────────────────────────────────────────
{cytotype_summary}

📦 S3 Output Artifacts:
──────────────────────────────────────────────────────────────
• Full Stomata Measurements (per-stoma):
  s3://{S3_BUCKET}/{s3_final_stoma}

• Image-Level Summary Table (per-image):
  s3://{S3_BUCKET}/{s3_final_summ}

Best regards,
Your Automated CESNET Stomata Pipeline
"""
    send_email_notification(subject, email_body)
    print("[+] Pipeline completely finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images (for testing)")
    args = parser.parse_args()
    run_full_inference(limit=args.limit)
