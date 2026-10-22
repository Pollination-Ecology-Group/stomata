#!/usr/bin/env python3
"""
Pipeline coordination helper for the CESNET k8s stomata pipeline.

Commands (first CLI argument):
  ARCHIVE  Copy the current inference results to
           inference_results/archive/<RUN_ID>_previous/ and remember which
           training run produced them. Run BEFORE training/inference so the
           report can compare old vs. new results.
  DONE     Write a success marker (JSON) with the previous and new training run
           folders and the dataset zip used. The report job waits for it.
  FAILED   Write a failure marker so the report job can e-mail a failure notice
           instead of waiting until its timeout.

Env: S3_ENDPOINT, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, RUN_ID,
     S3_OUTPUT_PREFIX, S3_RUNS_PREFIX, S3_STAGED_PREFIX
"""
import json
import os
import sys
from datetime import datetime, timezone

import boto3
from botocore.client import Config

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
RUN_ID = os.environ.get("RUN_ID", "manual")
OUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "Ostatni/Stomata/inference_results").rstrip("/")
RUNS_PREFIX = os.environ.get("S3_RUNS_PREFIX", "Ostatni/Stomata/training_runs/")
STAGED_PREFIX = os.environ.get("S3_STAGED_PREFIX", "Ostatni/Stomata/staged_area/")
PIPELINE_PREFIX = os.environ.get("S3_PIPELINE_PREFIX", "Ostatni/Stomata/pipeline_runs").rstrip("/")

RESULT_FILES = [
    "stomata_traits_all_3253_image_summary.csv",
    "stomata_traits_all_3253_stomata.csv",
]


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def latest_training_run(s3):
    """Return the newest training_runs/stomata_<timestamp>/ prefix (sorted by name)."""
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=RUNS_PREFIX, Delimiter="/")
    runs = sorted(p["Prefix"] for p in resp.get("CommonPrefixes", []))
    return runs[-1] if runs else None


def latest_staged_zip(s3):
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=STAGED_PREFIX)
    zips = [o for o in resp.get("Contents", []) if o["Key"].lower().endswith(".zip")]
    if not zips:
        return None
    return max(zips, key=lambda o: o["LastModified"])["Key"]


def put_json(s3, key, payload):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(payload, indent=2).encode())
    print(f"[+] Wrote s3://{S3_BUCKET}/{key}")


def cmd_archive(s3):
    dest = f"{OUT_PREFIX}/archive/{RUN_ID}_previous"
    for name in RESULT_FILES:
        src = f"{OUT_PREFIX}/{name}"
        try:
            s3.copy_object(Bucket=S3_BUCKET, Key=f"{dest}/{name}",
                           CopySource={"Bucket": S3_BUCKET, "Key": src})
            print(f"[+] Archived {src} -> {dest}/{name}")
        except Exception as e:  # first-ever run: nothing to archive
            print(f"[!] Could not archive {src}: {e}")
    put_json(s3, f"{PIPELINE_PREFIX}/{RUN_ID}/previous.json", {
        "run_id": RUN_ID,
        "previous_training_run": latest_training_run(s3),
        "previous_results_prefix": dest,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    })


def cmd_done(s3):
    put_json(s3, f"{PIPELINE_PREFIX}/{RUN_ID}/DONE.json", {
        "run_id": RUN_ID,
        "new_training_run": latest_training_run(s3),
        "new_results_prefix": OUT_PREFIX,
        "dataset_zip": latest_staged_zip(s3),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })


def cmd_failed(s3):
    put_json(s3, f"{PIPELINE_PREFIX}/{RUN_ID}/FAILED.json", {
        "run_id": RUN_ID,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"ARCHIVE", "DONE", "FAILED"}:
        sys.exit("usage: pipeline_marker.py ARCHIVE|DONE|FAILED")
    {"ARCHIVE": cmd_archive, "DONE": cmd_done, "FAILED": cmd_failed}[sys.argv[1]](s3_client())
