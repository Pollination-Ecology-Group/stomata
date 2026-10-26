#!/usr/bin/env python3
"""
app.py — Stomata Detection & Measurement Precision Review Server
Production Flask + Waitress application hosted on CESNET Kubernetes.

Allows researchers to visually inspect stoma segmentation and morphometric measurements,
use an interactive digital caliper to measure ground-truth dimensions,
grade measurement precision (High Precision / Minor Error / Poor / False Stoma),
flags crops or whole micrographs for Roboflow active learning relabeling,
exports Roboflow-ready segmentation zips, and shares authenticated access with Pollen Viability QC.
"""

import os
import io
import csv
import json
import time
import gzip
import hmac
import hashlib
import zipfile
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, Response, jsonify, request, send_from_directory, abort, g

HERE = Path(__file__).resolve().parent
BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
PREFIX = os.environ.get("REVIEW_PREFIX", "Ostatni/Stomata/Review").rstrip("/")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
URL_TTL = 12 * 3600

# Authentication: {"user": "pbkdf2$<iterations>$<salt hex>$<hash hex>"} from the K8s secret
USERS = json.loads(os.environ.get("REVIEW_USERS_JSON", "{}") or "{}")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"})
)

app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
lock = threading.Lock()

cache = {
    "manifest": None,
    "mtime": None,
    "loaded": 0.0,
    "decisions": {},
    "dec_loaded": 0.0,
    "relabel": {},
    "relabel_loaded": 0.0,
    "signed_urls": {},
    "signed_at": 0.0
}


def sanitize_nan(obj):
    if isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan(x) for x in obj]
    return obj


def check_password(user, pw):
    stored = USERS.get(user)
    if not stored or not pw:
        return False
    try:
        _, it, salt, digest = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(it)).hex()
        return hmac.compare_digest(calc, digest)
    except Exception:
        return False


@app.before_request
def authenticate():
    """Shared authentication with Pollen Viability QC app."""
    if request.path == "/healthz" or request.path.startswith("/api/img/") or request.path.startswith("/static/"):
        return None
    if not USERS:  # local dev mode
        name = request.args.get("reviewer", "").strip().lower()
        g.reviewer = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "jakub"
        return None
    auth = request.authorization
    user = (auth.username or "").strip().lower() if auth else ""
    if auth and check_password(user, auth.password):
        g.reviewer = user
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Stomata Precision QC"'})


def load_manifest():
    """Loads manifest.json from local file or falls back to S3, along with decisions and relabel queue."""
    local_manifest = HERE / "manifest.json"
    m = None
    mtime = 0

    if local_manifest.exists():
        try:
            with open(local_manifest, "r") as f:
                m = json.loads(f.read(), parse_constant=lambda c: None)
            mtime = os.path.getmtime(local_manifest)
        except Exception as e:
            print(f"[-] Error reading local manifest: {e}")

    if m is None:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")
            m = json.loads(obj["Body"].read().decode("utf-8"), parse_constant=lambda c: None)
            mtime = time.time()
        except Exception as e:
            print(f"[-] Error reading S3 manifest: {e}")
            m = {"items": [], "slides": [], "global_stats": {}}

    m = sanitize_nan(m)

    # 1. Fetch decisions from S3
    decisions = defaultdict(dict)
    try:
        pfx = f"{PREFIX}/decisions/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=pfx):
            for o in page.get("Contents", []):
                key = o["Key"]
                if key.endswith(".json"):
                    parts = key[len(pfx):].split("/")
                    if len(parts) >= 2:
                        item_id = parts[0]
                        rev_name = parts[1][:-5]
                        try:
                            body = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8"))
                            decisions[item_id][rev_name] = body
                        except Exception:
                            pass
    except Exception as e:
        print(f"[-] Warning: Failed to fetch decisions from S3: {e}")

    # 2. Fetch relabel queue from S3
    relabel_queue = {}
    try:
        pfx_rel = f"{PREFIX}/relabel_queue/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=pfx_rel):
            for o in page.get("Contents", []):
                key = o["Key"]
                if key.endswith(".json"):
                    item_id = key[len(pfx_rel):-5]
                    try:
                        body = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8"))
                        relabel_queue[item_id] = body
                    except Exception:
                        relabel_queue[item_id] = {"item_id": item_id}
    except Exception as e:
        print(f"[-] Warning: Failed to fetch relabel queue from S3: {e}")

    with lock:
        cache.update({
            "manifest": m,
            "mtime": mtime,
            "loaded": time.time(),
            "decisions": dict(decisions),
            "dec_loaded": time.time(),
            "relabel": relabel_queue,
            "relabel_loaded": time.time(),
            "signed_urls": {},
            "signed_at": 0.0
        })
    return m


def get_manifest():
    with lock:
        if cache["manifest"] is not None and (time.time() - cache["loaded"] < 120):
            return cache["manifest"]
    return load_manifest()


def get_decisions():
    with lock:
        if cache["decisions"] is not None and (time.time() - cache["dec_loaded"] < 30):
            return cache["decisions"]
    load_manifest()
    return cache["decisions"]


def get_relabel_queue():
    with lock:
        if cache.get("relabel") is not None and (time.time() - cache.get("relabel_loaded", 0) < 30):
            return cache["relabel"]
    load_manifest()
    return cache.get("relabel", {})


def sign_url(key):
    if not key:
        return None
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=URL_TTL
        )
    except Exception:
        return None


@app.get("/")
def index():
    return send_from_directory(str(HERE), "index.html")


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/api/img/<path:key>")
def stream_image(key):
    # Check local static file first
    local_file = HERE / key
    if local_file.exists():
        return send_from_directory(str(local_file.parent), local_file.name)
    # Otherwise stream from S3
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        mimetype = "image/jpeg"
        if key.lower().endswith(".png"):
            mimetype = "image/png"
        return Response(
            obj["Body"].read(),
            mimetype=mimetype,
            headers={"Cache-Control": "public, max-age=3600, must-revalidate"}
        )
    except Exception as e:
        print(f"[-] Error streaming image {key}: {e}")
        abort(404)


@app.get("/api/state")
def api_state():
    m = get_manifest()
    dec = get_decisions()
    rq = get_relabel_queue()
    current_reviewer = getattr(g, "reviewer", "jakub")

    items_payload = []
    for it in m.get("items", []):
        it_copy = dict(it)
        crop_s3 = it.get("crop_s3_key")
        # Serve via local proxy with cache buster for fresh display
        if crop_s3:
            it_copy["img_url"] = f"/api/img/{crop_s3}?v=clean3"
        else:
            it_copy["img_url"] = it.get("crop_url")
        items_payload.append(it_copy)

    slides_payload = []
    for sl in m.get("slides", []):
        sl_copy = dict(sl)
        ov_s3 = sl.get("overview_s3_key")
        if ov_s3:
            sl_copy["img_url"] = f"/api/img/{ov_s3}?v=clean3"
        else:
            sl_copy["img_url"] = sl.get("overview_url")
        slides_payload.append(sl_copy)

    return jsonify({
        "items": items_payload,
        "slides": slides_payload,
        "global_stats": m.get("global_stats", {}),
        "px_to_um": m.get("px_to_um", 50.0 / 451.3958),
        "decisions": dec,
        "relabel_items": list(rq.keys()),
        "relabel_count": len(rq),
        "current_user": current_reviewer
    })


@app.post("/api/decision")
def api_decision():
    data = request.get_json(force=True)
    item_id = str(data.get("id", "")).strip()
    rev_auth = getattr(g, "reviewer", None)
    reviewer = rev_auth if rev_auth else str(data.get("reviewer", "")).strip().lower()

    if not item_id or not reviewer or "/" in item_id or "/" in reviewer:
        return jsonify({"error": "Invalid item_id or reviewer handle"}), 400

    decision = {
        "verdict": data.get("verdict", "high_precision"),  # high_precision, minor_error, poor_measurement, false_stoma
        "precision_score": int(data.get("precision_score", 5)),
        "issues": list(data.get("issues", [])),
        "caliper": data.get("caliper", {}),
        "comment": str(data.get("comment", "")).strip(),
        "seconds": float(data.get("seconds", 0.0)),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reviewer": reviewer
    }

    # Save to S3
    s3_key = f"{PREFIX}/decisions/{item_id}/{reviewer}.json"
    try:
        s3.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=json.dumps(decision, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"[-] Error writing decision to S3: {e}")

    with lock:
        cache["decisions"].setdefault(item_id, {})[reviewer] = decision

    return jsonify({"ok": True, "saved": decision})


@app.post("/api/relabel")
def api_relabel():
    data = request.get_json(force=True)
    item_id = str(data.get("id", "")).strip()
    relabel_flag = bool(data.get("relabel", True))
    comment = str(data.get("comment", "")).strip()
    reviewer = getattr(g, "reviewer", "jakub")

    if not item_id or "/" in item_id:
        return jsonify({"error": "Invalid item id"}), 400

    s3_key = f"{PREFIX}/relabel_queue/{item_id}.json"
    if relabel_flag:
        payload = {
            "id": item_id,
            "reviewer": reviewer,
            "flagged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "comment": comment
        }
        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=s3_key,
                Body=json.dumps(payload, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
        except Exception as e:
            print(f"[-] Error writing relabel queue to S3: {e}")
        with lock:
            cache.setdefault("relabel", {})[item_id] = payload
    else:
        try:
            s3.delete_object(Bucket=BUCKET, Key=s3_key)
        except Exception as e:
            print(f"[-] Error removing relabel item from S3: {e}")
        with lock:
            cache.setdefault("relabel", {}).pop(item_id, None)

    return jsonify({"ok": True, "relabel_count": len(cache.get("relabel", {})), "is_relabel": relabel_flag})


@app.get("/api/relabel")
def api_get_relabel():
    rq = get_relabel_queue()
    return jsonify({
        "items": list(rq.keys()),
        "count": len(rq),
        "details": rq
    })


@app.get("/api/export/roboflow_relabel.zip")
def export_roboflow_relabel():
    m = get_manifest()
    rq = get_relabel_queue()
    if not rq:
        return Response("No crops flagged for relabeling yet.", status=400, mimetype="text/plain")

    items_by_id = {it["id"]: it for it in m.get("items", [])}
    slides_by_stem = {sl["stem"]: sl for sl in m.get("slides", [])}

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. data.yaml
        data_yaml = "names:\n- center\n- stoma\nnc: 2\n"
        zf.writestr("data.yaml", data_yaml)

        # 2. relabel_manifest.csv
        csv_buf = io.StringIO()
        csv_writer = csv.writer(csv_buf)
        csv_writer.writerow(["id", "filename", "type", "plant_ID", "cytotype", "leaf_side", "reviewer", "flagged_at", "comment"])

        for item_id, meta in rq.items():
            it = items_by_id.get(item_id)
            sl = slides_by_stem.get(item_id)

            if it:
                fn = it.get("filename", f"{item_id}.jpg")
                stem = Path(fn).stem
                csv_writer.writerow([
                    item_id, fn, "crop",
                    it.get("plant_ID", ""), it.get("cytotype", ""), it.get("leaf_side", ""),
                    meta.get("reviewer", ""), meta.get("flagged_at", ""), meta.get("comment", "")
                ])

                # Get image bytes
                img_bytes = None
                local_crop = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "images" / fn
                if local_crop.exists():
                    img_bytes = local_crop.read_bytes()
                elif it.get("crop_s3_key"):
                    try:
                        img_bytes = s3.get_object(Bucket=BUCKET, Key=it["crop_s3_key"])["Body"].read()
                    except Exception as e:
                        print(f"[-] Error fetching crop {fn} from S3: {e}")

                if img_bytes:
                    zf.writestr(f"train/images/{fn}", img_bytes)

                # Get YOLO segmentation label, filtering out degenerate border pores
                lbl_lines = []
                raw_lines = []
                local_lbl = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "labels" / f"{stem}.txt"
                if local_lbl.exists():
                    raw_lines = local_lbl.read_text().splitlines()
                else:
                    v = it.get("vectors", {})
                    W = float(it.get("crop_w", 640))
                    H = float(it.get("crop_h", 640))
                    # Class 0: pore
                    pore_poly = v.get("pore_poly")
                    if pore_poly and len(pore_poly) >= 3:
                        c_str = [f"{max(0.0, min(1.0, pt[0]/W)):.6f} {max(0.0, min(1.0, pt[1]/H)):.6f}" for pt in pore_poly]
                        raw_lines.append("0 " + " ".join(c_str))
                    # Class 1: stoma
                    stoma_poly = v.get("stoma_poly")
                    if stoma_poly and len(stoma_poly) >= 3:
                        c_str = [f"{max(0.0, min(1.0, pt[0]/W)):.6f} {max(0.0, min(1.0, pt[1]/H)):.6f}" for pt in stoma_poly]
                        raw_lines.append("1 " + " ".join(c_str))
                    # Companions
                    for cs in v.get("companion_stomata", []):
                        if cs.get("pore_poly") and len(cs["pore_poly"]) >= 3:
                            c_str = [f"{max(0.0, min(1.0, pt[0]/W)):.6f} {max(0.0, min(1.0, pt[1]/H)):.6f}" for pt in cs["pore_poly"]]
                            raw_lines.append("0 " + " ".join(c_str))
                        if cs.get("stoma_poly") and len(cs["stoma_poly"]) >= 3:
                            c_str = [f"{max(0.0, min(1.0, pt[0]/W)):.6f} {max(0.0, min(1.0, pt[1]/H)):.6f}" for pt in cs["stoma_poly"]]
                            raw_lines.append("1 " + " ".join(c_str))

                # Filter out degenerate border pore slivers
                for line in raw_lines:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        cls_id = parts[0]
                        coords = [float(x) for x in parts[1:]]
                        xs = coords[0::2]
                        ys = coords[1::2]
                        if cls_id == "0":
                            w = max(xs) - min(xs)
                            h = max(ys) - min(ys)
                            touches_edge = min(xs) <= 0.005 or min(ys) <= 0.005 or max(xs) >= 0.995 or max(ys) >= 0.995
                            if touches_edge and (w < 0.015 or h < 0.015):
                                continue  # skip degenerate border pore sliver
                        lbl_lines.append(line.strip())

                zf.writestr(f"train/labels/{stem}.txt", "\n".join(lbl_lines) + "\n")

            elif sl:
                fn = f"{sl['stem']}_overview.jpg"
                csv_writer.writerow([
                    item_id, fn, "micrograph",
                    sl.get("plant_ID", ""), "", sl.get("leaf_side", ""),
                    meta.get("reviewer", ""), meta.get("flagged_at", ""), meta.get("comment", "")
                ])
                local_ov = HERE / "static" / "overviews" / fn
                if local_ov.exists():
                    zf.writestr(f"train/images/{fn}", local_ov.read_bytes())
                elif sl.get("overview_s3_key"):
                    try:
                        img_bytes = s3.get_object(Bucket=BUCKET, Key=sl["overview_s3_key"])["Body"].read()
                        zf.writestr(f"train/images/{fn}", img_bytes)
                    except Exception as e:
                        print(f"[-] Error fetching slide {fn} from S3: {e}")

        zf.writestr("relabel_manifest.csv", csv_buf.getvalue())

    zip_bytes = zip_buf.getvalue()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Response(
        zip_bytes,
        mimetype="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=stomata_relabel_roboflow_{ts}.zip",
            "Content-Length": str(len(zip_bytes))
        }
    )


@app.get("/api/progress")
def api_progress():
    m = get_manifest()
    dec = get_decisions()
    items = m.get("items", [])
    total_items = len(items)

    verdicts_count = {"high_precision": 0, "minor_error": 0, "poor_measurement": 0, "false_stoma": 0}
    issue_counts = defaultdict(int)
    reviewers_stat = defaultdict(lambda: {"count": 0, "seconds": [], "last_active": ""})
    caliper_diffs_major = []
    caliper_diffs_minor = []
    caliper_diffs_pore = []

    item_lookup = {it["id"]: it for it in items}

    for item_id, per_reviewer in dec.items():
        it = item_lookup.get(item_id, {})
        for rev, d in per_reviewer.items():
            v = d.get("verdict", "high_precision")
            verdicts_count[v] = verdicts_count.get(v, 0) + 1

            for iss in d.get("issues", []):
                issue_counts[iss] += 1

            r_entry = reviewers_stat[rev]
            r_entry["count"] += 1
            if d.get("seconds"):
                r_entry["seconds"].append(d["seconds"])
            r_entry["last_active"] = max(r_entry["last_active"], d.get("saved_at", ""))

            # Caliper discrepancies
            cal = d.get("caliper", {})
            if cal and cal.get("measured") and it:
                if cal.get("major_axis_um") is not None and it.get("major_axis_um"):
                    diff = round(float(it["major_axis_um"]) - float(cal["major_axis_um"]), 2)
                    caliper_diffs_major.append(diff)
                if cal.get("minor_axis_um") is not None and it.get("minor_axis_um"):
                    diff = round(float(it["minor_axis_um"]) - float(cal["minor_axis_um"]), 2)
                    caliper_diffs_minor.append(diff)
                if cal.get("pore_width_um") is not None and it.get("pore_width_um"):
                    diff = round(float(it["pore_width_um"]) - float(cal["pore_width_um"]), 2)
                    caliper_diffs_pore.append(diff)

    done_items = len([it_id for it_id, d in dec.items() if len(d) > 0])

    def calc_error_stats(diffs):
        if not diffs:
            return {"count": 0, "mae_um": None, "mean_bias_um": None, "max_err_um": None}
        abs_diffs = [abs(x) for x in diffs]
        return {
            "count": len(diffs),
            "mae_um": round(sum(abs_diffs) / len(abs_diffs), 2),
            "mean_bias_um": round(sum(diffs) / len(diffs), 2),
            "max_err_um": round(max(abs_diffs), 2)
        }

    return jsonify({
        "total_items": total_items,
        "done_items": done_items,
        "completion_pct": round(done_items / max(1, total_items) * 100, 1),
        "verdicts": verdicts_count,
        "issues": dict(issue_counts),
        "reviewers": {k: {"count": v["count"], "last_active": v["last_active"],
                          "median_sec": round(float(np.median(v["seconds"])), 1) if v["seconds"] else None}
                      for k, v in reviewers_stat.items()},
        "precision_benchmarks": {
            "major_axis": calc_error_stats(caliper_diffs_major),
            "minor_axis": calc_error_stats(caliper_diffs_minor),
            "pore_width": calc_error_stats(caliper_diffs_pore)
        }
    })


@app.get("/api/export/precision.csv")
def export_precision():
    m = get_manifest()
    dec = get_decisions()
    items_by_id = {it["id"]: it for it in m.get("items", [])}

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "item_id", "image_name", "stoma_idx", "plant_ID", "cytotype", "ploidy", "leaf_side",
        "reviewer", "verdict", "precision_score", "issues",
        "model_major_axis_um", "caliper_major_axis_um", "delta_major_axis_um", "err_pct_major",
        "model_minor_axis_um", "caliper_minor_axis_um", "delta_minor_axis_um", "err_pct_minor",
        "model_pore_width_um", "caliper_pore_width_um", "delta_pore_width_um",
        "focus_score", "is_suitable_for_ploidy_sizing", "comment", "seconds", "saved_at"
    ])

    for item_id, per_reviewer in dec.items():
        it = items_by_id.get(item_id, {})
        for rev, d in per_reviewer.items():
            cal = d.get("caliper", {})

            m_maj = it.get("major_axis_um", "")
            c_maj = cal.get("major_axis_um", "")
            d_maj = round(float(m_maj) - float(c_maj), 2) if m_maj != "" and c_maj != "" else ""
            err_maj = round(abs(float(d_maj)) / float(c_maj) * 100, 1) if d_maj != "" and float(c_maj) > 0 else ""

            m_min = it.get("minor_axis_um", "")
            c_min = cal.get("minor_axis_um", "")
            d_min = round(float(m_min) - float(c_min), 2) if m_min != "" and c_min != "" else ""
            err_min = round(abs(float(d_min)) / float(c_min) * 100, 1) if d_min != "" and float(c_min) > 0 else ""

            m_por = it.get("pore_width_um", "")
            c_por = cal.get("pore_width_um", "")
            d_por = round(float(m_por) - float(c_por), 2) if m_por != "" and c_por != "" else ""

            writer.writerow([
                item_id, it.get("image_name", ""), it.get("stoma_idx", ""),
                it.get("plant_ID", ""), it.get("cytotype", ""), it.get("ploidy", ""), it.get("leaf_side", ""),
                rev, d.get("verdict", ""), d.get("precision_score", ""), ";".join(d.get("issues", [])),
                m_maj, c_maj, d_maj, err_maj,
                m_min, c_min, d_min, err_min,
                m_por, c_por, d_por,
                it.get("focus_score", ""), it.get("is_suitable_for_ploidy_sizing", ""),
                d.get("comment", ""), d.get("seconds", ""), d.get("saved_at", "")
            ])

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=stomata_measurement_precision.csv"}
    )


@app.after_request
def compress_response(resp):
    if (resp.mimetype in ("application/json", "text/csv") and
            resp.status_code == 200 and
            "gzip" in request.headers.get("Accept-Encoding", "") and
            len(resp.get_data()) > 2048):
        resp.set_data(gzip.compress(resp.get_data(), compresslevel=5))
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Vary"] = "Accept-Encoding"
    return resp


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", "8080"))
    print(f"🚀 Stomata Measurement Precision App serving on http://0.0.0.0:{port}")
    serve(app, host="0.0.0.0", port=port, threads=8)
