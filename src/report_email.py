#!/usr/bin/env python3
"""
Report job, final step: e-mail the rendered PDF report (or a failure notice)
and upload the PDF to S3.

Reads WORK_DIR/status.txt (DONE/FAILED/TIMEOUT), WORK_DIR/run_info.json,
WORK_DIR/short_report.pdf and WORK_DIR/render.log.
"""
import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import boto3
from botocore.client import Config

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
RUN_ID = os.environ.get("RUN_ID", "manual")
REPORTS_PREFIX = os.environ.get("S3_REPORTS_PREFIX", "Ostatni/Stomata/reports").rstrip("/")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/work"))
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT", "jakubstenc@gmail.com")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() in ("1", "true", "yes")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def tail(path: Path, n=60):
    if not path.exists():
        return "(no log)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def main():
    status = (WORK_DIR / "status.txt").read_text().strip() if (WORK_DIR / "status.txt").exists() else "UNKNOWN"
    info = json.loads((WORK_DIR / "run_info.json").read_text()) if (WORK_DIR / "run_info.json").exists() else {}
    pdf = WORK_DIR / "short_report.pdf"
    render_log = WORK_DIR / "render.log"

    msg = EmailMessage()
    msg["From"] = EMAIL_RECIPIENT
    msg["To"] = EMAIL_RECIPIENT

    s3_pdf_key = None
    if status == "DONE" and pdf.exists():
        s3_pdf_key = f"{REPORTS_PREFIX}/short_report_{RUN_ID}.pdf"
        try:
            s3_client().upload_file(str(pdf), S3_BUCKET, s3_pdf_key)
            print(f"[+] Uploaded report to s3://{S3_BUCKET}/{s3_pdf_key}")
        except Exception as e:
            print(f"[!] Report upload failed: {e}")
        msg["Subject"] = f"Stomata pipeline {RUN_ID}: new results + report (PDF attached)"
        msg.set_content(
            "Hello Jakub,\n\n"
            "Retraining, full inference and the short report finished on the CESNET cluster.\n\n"
            f"Dataset:            {info.get('dataset_zip')}\n"
            f"New training run:   {info.get('new_training_run')}\n"
            f"Previous run:       {info.get('previous_training_run')}\n"
            f"New results:        s3://{S3_BUCKET}/{info.get('new_results_prefix')}/\n"
            f"Report on S3:       s3://{S3_BUCKET}/{s3_pdf_key}\n\n"
            "The attached PDF includes a comparison with the previous model's results.\n\n"
            "CESNET Automated Stomata Pipeline\n"
        )
        msg.add_attachment(pdf.read_bytes(), maintype="application", subtype="pdf",
                           filename=f"stomata_short_report_{RUN_ID}.pdf")
    elif status == "DONE":
        msg["Subject"] = f"Stomata pipeline {RUN_ID}: results ready, but report rendering FAILED"
        msg.set_content(
            "Training and inference finished, but the PDF report could not be rendered.\n"
            f"New results: s3://{S3_BUCKET}/{info.get('new_results_prefix')}/\n\n"
            f"Last lines of render log:\n{tail(render_log)}\n"
        )
    else:
        msg["Subject"] = f"Stomata pipeline {RUN_ID}: {status}"
        msg.set_content(
            f"The training/inference job did not finish successfully (status: {status}).\n"
            "Check: kubectl logs job/stomata-train-job -n stenc-ns\n"
        )

    body = msg.get_body(preferencelist=("plain",))
    print(f"Subject: {msg['Subject']}\n\n{body.get_content() if body else ''}")
    if DRY_RUN:
        print("[*] DRY_RUN set: not sending e-mail.")
        return
    pw = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not pw:
        print("[!] GMAIL_APP_PASSWORD not set; cannot send e-mail.")
        return
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(EMAIL_RECIPIENT, pw)
        server.send_message(msg)
    print(f"[+] E-mail sent to {EMAIL_RECIPIENT}")


if __name__ == "__main__":
    main()
