#!/usr/bin/env python3
"""
Report job, step 1: wait for the train/inference job to finish, then download
everything the short report needs into WORK_DIR.

Writes WORK_DIR/status.txt with one of: DONE, FAILED, TIMEOUT
and WORK_DIR/run_info.json with run metadata (training runs, dataset zip).
Always exits 0 so that the e-mail step runs in every case.
"""
import json
import os
import time
from pathlib import Path

import boto3
from botocore.client import Config

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
RUN_ID = os.environ.get("RUN_ID", "manual")
PIPELINE_PREFIX = os.environ.get("S3_PIPELINE_PREFIX", "Ostatni/Stomata/pipeline_runs").rstrip("/")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/work"))
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "300"))
MAX_WAIT_HOURS = float(os.environ.get("MAX_WAIT_HOURS", "14"))

SUMMARY = "stomata_traits_all_3253_image_summary.csv"
STOMATA = "stomata_traits_all_3253_stomata.csv"


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def get_json(s3, key):
    try:
        return json.loads(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        if "NoSuchKey" in str(e) or "404" in str(e):
            return None
        raise


def download(s3, key, dest: Path):
    try:
        s3.download_file(S3_BUCKET, key, str(dest))
        print(f"[+] {key} -> {dest} ({dest.stat().st_size:,} bytes)", flush=True)
        return True
    except Exception as e:
        print(f"[!] Could not download {key}: {e}", flush=True)
        return False


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    s3 = s3_client()
    base = f"{PIPELINE_PREFIX}/{RUN_ID}"
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    print(f"[*] Waiting for s3://{S3_BUCKET}/{base}/DONE.json (max {MAX_WAIT_HOURS} h)", flush=True)

    status, done = "TIMEOUT", None
    while time.time() < deadline:
        done = get_json(s3, f"{base}/DONE.json")
        if done:
            status = "DONE"
            break
        if get_json(s3, f"{base}/FAILED.json"):
            status = "FAILED"
            break
        time.sleep(POLL_SECONDS)
    print(f"[*] Pipeline status: {status}", flush=True)

    info = {"run_id": RUN_ID, "status": status}
    if status == "DONE":
        prev = get_json(s3, f"{base}/previous.json") or {}
        info.update(done)
        info["previous_training_run"] = prev.get("previous_training_run")
        new_res = done["new_results_prefix"].rstrip("/")
        download(s3, f"{new_res}/{SUMMARY}", WORK_DIR / "new_image_summary.csv")
        download(s3, f"{new_res}/{STOMATA}", WORK_DIR / "new_stomata.csv")
        if prev.get("previous_results_prefix"):
            p = prev["previous_results_prefix"].rstrip("/")
            download(s3, f"{p}/{SUMMARY}", WORK_DIR / "prev_image_summary.csv")
            download(s3, f"{p}/{STOMATA}", WORK_DIR / "prev_stomata.csv")
        if done.get("new_training_run"):
            download(s3, f"{done['new_training_run']}results.csv", WORK_DIR / "new_train_results.csv")
        if prev.get("previous_training_run") and prev["previous_training_run"] != done.get("new_training_run"):
            download(s3, f"{prev['previous_training_run']}results.csv", WORK_DIR / "prev_train_results.csv")

    (WORK_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    (WORK_DIR / "status.txt").write_text(status)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
