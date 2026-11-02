#!/usr/bin/env python3
"""
app.py — Stomata Detection QC, Rapid Evaluation & Active Learning Label Studio
Hosted on CESNET Kubernetes (stomata-review-stenc.dyn.cloud.e-infra.cz).

Two core operating modes:
1. Fast Detection Evaluation:
   - High-throughput verification of YOLO segmentation masks & morphometrics.
   - 1-click & hotkey grading (High Precision / Minor Error / Poor Outline / False Stoma).
   - Live model accuracy & precision benchmarks.
   - Automatically enqueues poor outlines & false detections into the Relabeling Studio.

2. In-App Relabeling Studio:
   - Eliminates Roboflow completely: annotate & refine stoma/pore polygons directly in-app.
   - Active Learning Priority Queue: items ordered by what will improve the model most first
     (human eval failures, low confidence, missing pore aperture, border cuts, rare cytotypes).
   - Interactive polygon vertex editing, polygon drawing, box/ellipse fitting.
   - Direct YOLO segmentation format (.txt) persistence & S3 dataset staging for train_stomata_s3.py.
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
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import numpy as np
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, Response, jsonify, request, send_from_directory, abort, g, send_file

HERE = Path("/app") if Path("/app/index.html").exists() else Path(__file__).parent
DATA_DIR = Path(os.environ.get("STOMATA_DATA_DIR", "/tmp/stomata_data" if Path("/app/index.html").exists() else HERE / "data"))
LABELS_DIR = DATA_DIR / "labels"
DECISIONS_DIR = DATA_DIR / "decisions"
EXPORTS_DIR = DATA_DIR / "exports"
for _d in (DATA_DIR, LABELS_DIR, DECISIONS_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BUCKET = os.environ.get("S3_BUCKET", "peg-bucket")
PREFIX = os.environ.get("REVIEW_PREFIX", "Ostatni/Stomata/Review").rstrip("/")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
URL_TTL = 12 * 3600

# Authentication: {"user": "pbkdf2$<iterations>$<salt hex>$<hash hex>"} from the K8s secret
USERS = json.loads(os.environ.get("REVIEW_USERS_JSON", "{}") or "{}")

# S3 credentials fallback from environment or local Kubernetes secret
_ak = os.environ.get("AWS_ACCESS_KEY_ID")
_sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
if not _ak or not _sk:
    try:
        import subprocess, base64
        _out = subprocess.check_output(
            ["kubectl", "get", "secret", "s3-credentials", "-n", "stenc-ns", "-o", "json"],
            timeout=2, stderr=subprocess.DEVNULL
        )
        _sec = json.loads(_out)["data"]
        _ak = base64.b64decode(_sec["access-key"]).decode().strip()
        _sk = base64.b64decode(_sec["secret-key"]).decode().strip()
    except Exception:
        pass

s3 = None
if _ak and _sk:
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=_ak,
            aws_secret_access_key=_sk,
            verify=False,
            config=Config(signature_version="s3v4", connect_timeout=1.5, read_timeout=2.0, retries={"max_attempts": 1, "mode": "standard"})
        )
    except Exception as e:
        print(f"[-] S3 init warning: {e}")

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
    "labels": {},
    "labels_loaded": 0.0,
    "signed_urls": {},
    "signed_at": 0.0
}


# ── Segment Anything Model (SAM) Manager ─────────────────────────────────────
class SamManager:
    """Manages MobileSAM / SAM model for interactive prompt-assisted segmentation."""
    def __init__(self):
        self.model = None
        self.ready = False
        self.lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            candidates = [
                os.environ.get("SAM_WEIGHTS", ""),
                str(HERE / "mobile_sam.pt"),
                "/tmp/mobile_sam.pt",
                "/home/meow/Documents/Antigravity/YOLO_flies/mobile_sam.pt",
                str(Path(__file__).resolve().parent / "mobile_sam.pt")
            ]
            weights_path = None
            for c in candidates:
                if c and os.path.exists(c):
                    weights_path = c
                    break

            if not weights_path and s3 is not None:
                tmp_target = "/tmp/mobile_sam.pt"
                try:
                    print("[SAM] Downloading mobile_sam.pt from S3...")
                    s3.download_file(BUCKET, "Ostatni/YOLO_flies/mobile_sam.pt", tmp_target)
                    if os.path.exists(tmp_target):
                        weights_path = tmp_target
                except Exception as e:
                    print(f"[SAM] S3 download error: {e}")

            if weights_path and os.path.exists(weights_path):
                from ultralytics import SAM
                self.model = SAM(weights_path)
                self.ready = True
                print(f"🪄 [SAM] MobileSAM loaded successfully from {weights_path}")
            else:
                print("⚠️ [SAM] MobileSAM weights not found locally or on S3.")
        except Exception as e:
            print(f"⚠️ [SAM] Initialization warning: {e}")

    def segment(self, img_array, bbox=None, points=None, labels=None):
        if not self.ready or not self.model:
            return None
        with self.lock:
            kw = {}
            if bbox:
                kw["bboxes"] = [bbox]
            if points:
                kw["points"] = [points]
                kw["labels"] = [labels if labels else [1] * len(points)]
            
            res = self.model(img_array, verbose=False, **kw)
            if not res or res[0].masks is None or len(res[0].masks.xy) == 0:
                return None
            
            poly = res[0].masks.xy[0].tolist()
            # Downsample if overly dense (>100 points) to keep frontend smooth
            if len(poly) > 100:
                step = max(1, len(poly) // 75)
                poly = poly[::step]
            return poly


sam_mgr = SamManager()


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


def compute_active_learning_priority(item, decisions, relabel_queue, ground_truth_labels):
    """
    Computes an Active Learning Model Improvement Priority Score (IPS) and explanatory tags.
    Orders the relabeling queue by what will improve the YOLO segmentation model the most first:
      1. Human evaluation failures (Poor outline, False stoma, Relabel queue)
      2. Model uncertainty (Borderline / low YOLO confidence)
      3. Structural anomalies (Missing pore aperture, abnormal aspect ratio)
      4. Border / edge cutoffs
      5. Multi-stoma clutter
      6. Underrepresented cytotypes (neo_4x, nat_4x)
      7. High visual focus clarity (where human annotation yields high SNR)
    """
    item_id = item.get("id", "")
    dec_dict = decisions.get(item_id, {})
    is_relabel = item_id in relabel_queue
    saved_gt = ground_truth_labels.get(item_id)

    score = 0.0
    tags = []

    # 1. Human Evaluation Feedback (Tier 1 - Highest Impact)
    has_poor = any(d.get("verdict") == "poor_measurement" for d in dec_dict.values())
    has_false = any(d.get("verdict") == "false_stoma" for d in dec_dict.values())
    has_minor = any(d.get("verdict") == "minor_error" for d in dec_dict.values())
    has_immature = any("immature_no_pore" in d.get("issues", []) for d in dec_dict.values())

    if has_false:
        score += 150.0
        tags.append("⚠️ Flagged: False Stoma (Hard Negative)")
    elif has_poor:
        score += 140.0
        tags.append("⚠️ Flagged: Poor Outline")
    elif has_immature:
        score += 130.0
        tags.append("🌱 Flagged: Immature / No Pore")
    elif is_relabel:
        score += 120.0
        tags.append("🏷️ Queued for Relabel")
    elif has_minor:
        score += 65.0
        tags.append("⚠️ Minor Outline Error")

    # 2. Model Uncertainty / Confidence Boundary (Tier 2)
    stoma_conf = float(item.get("stoma_conf") or 0.0)
    center_conf = float(item.get("center_conf") or 0.0)

    if 0.0 < stoma_conf < 0.45:
        score += 70.0
        tags.append(f"⚠️ Low Conf ({stoma_conf:.2f})")
    elif 0.45 <= stoma_conf < 0.65:
        score += 45.0
        tags.append(f"🎲 Borderline Conf ({stoma_conf:.2f})")
    elif stoma_conf == 0.0 and not item.get("is_background_control"):
        score += 60.0
        tags.append("⚠️ Zero Conf Detection")

    if stoma_conf > 0 and center_conf > 0 and abs(stoma_conf - center_conf) > 0.30:
        score += 25.0
        tags.append("⚡ Conf Discrepancy")

    # 3. Morphometric & Structural Anomalies (Tier 3)
    vectors = item.get("vectors") or {}
    stoma_poly = vectors.get("stoma_poly") or []
    pore_poly = vectors.get("pore_poly") or []
    has_stoma_poly = len(stoma_poly) >= 3
    has_pore_poly = len(pore_poly) >= 3

    if has_stoma_poly and not has_pore_poly:
        score += 60.0
        tags.append("🔬 Missing Pore Aperture")
    elif has_pore_poly and not has_stoma_poly:
        score += 60.0
        tags.append("⚠️ Missing Stoma Boundary")

    aspect_ratio = float(item.get("aspect_ratio") or 0.0)
    if aspect_ratio > 2.8:
        score += 35.0
        tags.append(f"📏 Outlier Aspect ({aspect_ratio:.1f})")
    elif 0.0 < aspect_ratio < 0.95:
        score += 35.0
        tags.append(f"📏 Inverted Aspect ({aspect_ratio:.1f})")

    # 4. Border Truncations (Tier 4)
    if stoma_poly:
        xs = [p[0] for p in stoma_poly]
        ys = [p[1] for p in stoma_poly]
        w = float(item.get("crop_w") or 640)
        h = float(item.get("crop_h") or 640)
        if min(xs) <= 6 or min(ys) <= 6 or max(xs) >= w - 6 or max(ys) >= h - 6:
            score += 30.0
            tags.append("📐 Border Cut")

    # 5. Multi-Stoma Density (Tier 5)
    total_stomata = int(item.get("total_stomata_in_crop") or 1)
    if total_stomata >= 2:
        score += 20.0
        tags.append(f"👥 Multi-Stomata ({total_stomata})")

    # 6. Cytotype Diversity (Tier 6)
    cyto = str(item.get("cytotype") or "")
    if cyto in ("neo_4x", "nat_4x"):
        score += 25.0
        tags.append(f"🧬 Rare Cytotype: {cyto}")

    # 7. Focus Quality Bonus
    focus = float(item.get("focus_score") or 0.0)
    if item.get("is_well_focused") or focus >= 15.0:
        score += 15.0
        tags.append("✨ Sharp Focus")

    # 8. Saved Ground Truth penalty (completed items move to bottom of active learning queue)
    if saved_gt:
        score -= 300.0
        tags = ["✅ Ground Truth Saved"] + [t for t in tags if not t.startswith("✅")]

    return round(score, 1), tags


def load_manifest():
    """Loads manifest.json, decisions, relabel queue, and ground truth labels."""
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

    if m is None and s3 is not None:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/manifest.json")
            m = json.loads(obj["Body"].read().decode("utf-8"), parse_constant=lambda c: None)
            mtime = time.time()
        except Exception as e:
            print(f"[-] Error reading S3 manifest: {e}")

    if m is None:
        m = {"items": [], "slides": [], "global_stats": {}}

    m = sanitize_nan(m)

    # 1. Fetch decisions (S3 + local)
    decisions = defaultdict(dict)
    # Check local decisions directory
    if DECISIONS_DIR.exists():
        for p in DECISIONS_DIR.glob("*/*.json"):
            try:
                item_id = p.parent.name
                reviewer = p.stem
                decisions[item_id][reviewer] = json.loads(p.read_text())
            except Exception:
                pass

    if s3 is not None:
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

    # 2. Fetch relabel queue (S3 + local)
    relabel_queue = {}
    local_rq_file = DATA_DIR / "relabel_queue.json"
    if local_rq_file.exists():
        try:
            relabel_queue = json.loads(local_rq_file.read_text())
        except Exception:
            pass

    if s3 is not None:
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

    # 3. Fetch in-app ground-truth labels (S3 + local)
    ground_truth = {}
    if LABELS_DIR.exists():
        for p in LABELS_DIR.glob("*.json"):
            try:
                ground_truth[p.stem] = json.loads(p.read_text())
            except Exception:
                pass

    if s3 is not None:
        try:
            pfx_gt = f"{PREFIX}/ground_truth_labels/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=BUCKET, Prefix=pfx_gt):
                for o in page.get("Contents", []):
                    key = o["Key"]
                    if key.endswith(".json"):
                        item_id = key[len(pfx_gt):-5]
                        try:
                            body = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8"))
                            ground_truth[item_id] = body
                        except Exception:
                            pass
        except Exception as e:
            print(f"[-] Warning: Failed to fetch ground truth labels from S3: {e}")

    with lock:
        cache.update({
            "manifest": m,
            "mtime": mtime,
            "loaded": time.time(),
            "decisions": dict(decisions),
            "dec_loaded": time.time(),
            "relabel": relabel_queue,
            "relabel_loaded": time.time(),
            "labels": ground_truth,
            "labels_loaded": time.time(),
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


def get_labels():
    with lock:
        if cache.get("labels") is not None and (time.time() - cache.get("labels_loaded", 0) < 30):
            return cache["labels"]
    load_manifest()
    return cache.get("labels", {})


def sign_url(key):
    if not key or s3 is None:
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
    # Check local static file or roboflow dataset images first
    local_file = HERE / key
    if local_file.exists():
        return send_from_directory(str(local_file.parent), local_file.name)

    # Check roboflow preannotated folder
    alt_file = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "images" / Path(key).name
    if alt_file.exists():
        return send_from_directory(str(alt_file.parent), alt_file.name)

    # Otherwise stream from S3
    if s3 is not None:
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
            print(f"[-] Error streaming image {key} from S3: {e}")

    abort(404)


@app.get("/static/crops/<path:filename>")
def stream_crop_static(filename):
    local_file = HERE / "static" / "crops" / filename
    if local_file.exists():
        return send_from_directory(str(local_file.parent), local_file.name)
    alt_file = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "images" / Path(filename).name
    if alt_file.exists():
        return send_from_directory(str(alt_file.parent), alt_file.name)
    return stream_image(f"Ostatni/Stomata/Review/crops/{filename}")


@app.get("/static/overviews/<path:filename>")
def stream_overview_static(filename):
    local_file = HERE / "static" / "overviews" / filename
    if local_file.exists():
        return send_from_directory(str(local_file.parent), local_file.name)
    return stream_image(f"Ostatni/Stomata/Review/overviews/{filename}")


@app.get("/api/sam/status")
def api_sam_status():
    return jsonify({"ready": sam_mgr.ready})


@app.post("/api/sam/<item_id>")
def api_sam_segment(item_id):
    """
    Runs prompt-assisted SAM (Segment Anything Model) on the requested crop.
    Accepts bbox prompt [x1, y1, x2, y2] or point prompts [[x, y], ...].
    Returns the predicted polygon for either class 1 (stoma) or class 0 (pore).
    """
    if not sam_mgr.ready:
        return jsonify({"ok": False, "error": "SAM model is not loaded yet"}), 503

    data = request.get_json(force=True)
    bbox = data.get("bbox")      # [x1, y1, x2, y2]
    points = data.get("points")  # [[x, y], ...]
    labels = data.get("labels")  # [1, ...]
    cls_idx = int(data.get("cls", 1))

    m = get_manifest()
    item = next((it for it in m.get("items", []) if it.get("id") == item_id), None)
    if not item:
        return jsonify({"ok": False, "error": "Item not found"}), 404

    crop_s3 = item.get("crop_s3_key")
    crop_url = item.get("crop_url")
    img_bytes = None

    # Check local files first
    for cand in [
        HERE / (crop_s3 or ""),
        HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "images" / (Path(crop_s3).name if crop_s3 else ""),
        HERE.parent.parent / "roboflow_preannotated_dataset" / "images" / (Path(crop_s3).name if crop_s3 else ""),
        DATA_DIR / "crops" / (Path(crop_s3).name if crop_s3 else "")
    ]:
        if cand and cand.is_file():
            img_bytes = cand.read_bytes()
            break

    # Stream from S3 if needed
    if not img_bytes and s3 is not None and crop_s3:
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=crop_s3)
            img_bytes = obj["Body"].read()
        except Exception as e:
            print(f"[SAM] S3 fetch error for {crop_s3}: {e}")

    if not img_bytes:
        return jsonify({"ok": False, "error": "Crop image not accessible"}), 404

    from PIL import Image
    import numpy as np
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    poly = sam_mgr.segment(img, bbox=bbox, points=points, labels=labels)
    if not poly or len(poly) < 3:
        return jsonify({"ok": False, "error": "SAM could not extract a valid polygon"}), 400

    return jsonify({
        "ok": True,
        "item_id": item_id,
        "cls": cls_idx,
        "polygon": poly,
        "points_count": len(poly)
    })


@app.get("/api/state")
def api_state():
    m = get_manifest()
    dec = get_decisions()
    rq = get_relabel_queue()
    gt_labels = get_labels()
    current_reviewer = getattr(g, "reviewer", "jakub")

    # 1. Compute Active Learning Priority for all items
    raw_items = m.get("items", [])
    priorities = []
    for it in raw_items:
        score, tags = compute_active_learning_priority(it, dec, rq, gt_labels)
        priorities.append((it["id"], score, tags))

    # Rank highest score first
    priorities.sort(key=lambda x: x[1], reverse=True)
    rank_lookup = {pid: (rank + 1, score, tags) for rank, (pid, score, tags) in enumerate(priorities)}

    items_payload = []
    for it in raw_items:
        it_copy = dict(it)
        crop_s3 = it.get("crop_s3_key")
        if crop_s3:
            it_copy["img_url"] = f"/api/img/{crop_s3}?v=clean3"
        else:
            it_copy["img_url"] = it.get("crop_url")

        item_id = it.get("id")
        rank_info = rank_lookup.get(item_id, (999, 0.0, []))
        it_copy["priority_rank"] = rank_info[0]
        it_copy["priority_score"] = rank_info[1]
        it_copy["priority_tags"] = rank_info[2]

        # Ground truth status
        saved_gt = gt_labels.get(item_id)
        it_copy["is_ground_truth"] = bool(saved_gt)
        it_copy["ground_truth"] = saved_gt
        if saved_gt and saved_gt.get("objects"):
            v_synced = dict(it_copy.get("vectors") or {})
            for obj in saved_gt["objects"]:
                if obj.get("cls") == 1:
                    v_synced["stoma_poly"] = obj.get("polygon")
                elif obj.get("cls") == 0:
                    v_synced["pore_poly"] = obj.get("polygon")
            it_copy["vectors"] = v_synced

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

    # Priority Summary Statistics
    high_priority_count = len([it for it in items_payload if it["priority_score"] >= 80.0])
    flagged_failures = len([it for it in items_payload if any("Flagged" in t for t in it.get("priority_tags", []))])
    total_eval = len([iid for iid, d in dec.items() if len(d) > 0])
    high_prec_count = 0
    for iid, d in dec.items():
        if any(rev_d.get("verdict") == "high_precision" for rev_d in d.values()):
            high_prec_count += 1

    return jsonify({
        "items": items_payload,
        "slides": slides_payload,
        "global_stats": m.get("global_stats", {}),
        "px_to_um": m.get("px_to_um", 50.0 / 451.3958),
        "decisions": dec,
        "relabel_items": list(rq.keys()),
        "relabel_count": len(rq),
        "ground_truth_count": len(gt_labels),
        "current_user": current_reviewer,
        "priority_summary": {
            "total_items": len(items_payload),
            "high_priority_count": high_priority_count,
            "flagged_failures_count": flagged_failures,
            "ground_truth_count": len(gt_labels),
            "evaluation_done_count": total_eval,
            "evaluation_high_prec_pct": round(high_prec_count / max(1, total_eval) * 100.0, 1) if total_eval else 0.0
        }
    })


@app.post("/api/decision")
def api_decision():
    """Saves Fast Evaluation decision (High Precision, Minor Error, Poor Outline, False Stoma)."""
    data = request.get_json(force=True)
    item_id = str(data.get("id", "")).strip()
    rev_auth = getattr(g, "reviewer", None)
    reviewer = rev_auth if rev_auth else str(data.get("reviewer", "")).strip().lower()

    if not item_id or not reviewer or "/" in item_id or "/" in reviewer:
        return jsonify({"error": "Invalid item_id or reviewer handle"}), 400

    verdict = data.get("verdict", "high_precision")
    issues = list(data.get("issues", []))
    is_immature = "immature_no_pore" in issues
    is_suitable = False if is_immature else (verdict in ("high_precision", "minor_error"))
    counts_density = verdict != "false_stoma"

    decision = {
        "verdict": verdict,  # high_precision, minor_error, poor_measurement, false_stoma
        "precision_score": int(data.get("precision_score", 5)),
        "issues": issues,
        "is_suitable_for_ploidy_sizing": is_suitable,
        "counts_for_density": counts_density,
        "caliper": data.get("caliper", {}),
        "edited_vectors": data.get("edited_vectors", {}),
        "comment": str(data.get("comment", "")).strip(),
        "seconds": float(data.get("seconds", 0.0)),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reviewer": reviewer
    }

    # 1. Save locally
    item_dec_dir = DECISIONS_DIR / item_id
    item_dec_dir.mkdir(parents=True, exist_ok=True)
    (item_dec_dir / f"{reviewer}.json").write_text(json.dumps(decision, indent=2))

    # 2. Save to S3
    if s3 is not None:
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

    # 3. If verdict indicates failure or immature stoma, auto-queue into Relabeling Studio with high priority
    if verdict in ("poor_measurement", "false_stoma") or is_immature:
        rq_payload = {
            "id": item_id,
            "reviewer": reviewer,
            "flagged_at": decision["saved_at"],
            "comment": f"Auto-flagged from Detection Evaluation ({'Immature / No Pore' if is_immature else verdict})"
        }
        (DATA_DIR / "relabel_queue.json").write_text(json.dumps(cache.get("relabel", {}), indent=2))
        if s3 is not None:
            try:
                s3.put_object(
                    Bucket=BUCKET,
                    Key=f"{PREFIX}/relabel_queue/{item_id}.json",
                    Body=json.dumps(rq_payload, indent=2).encode("utf-8"),
                    ContentType="application/json"
                )
            except Exception:
                pass
        with lock:
            cache.setdefault("relabel", {})[item_id] = rq_payload

    with lock:
        cache["decisions"].setdefault(item_id, {})[reviewer] = decision

    return jsonify({"ok": True, "saved": decision})


@app.get("/api/label/<item_id>")
def api_get_label(item_id):
    """
    Returns ground-truth segmentation masks for an item in In-App Relabeling Studio.
    If already edited & saved, returns saved ground truth. Otherwise, synthesizes
    editable objects from model vectors so the user can easily adjust vertices.
    """
    m = get_manifest()
    labels = get_labels()

    if item_id in labels:
        lbl = labels[item_id]
        return jsonify({"ok": True, "item_id": item_id, "label": lbl, "objects": lbl.get("objects", []), "is_saved": True})

    # Find item in manifest and synthesize initial objects
    item = next((it for it in m.get("items", []) if it.get("id") == item_id), None)
    if not item:
        return jsonify({"error": "Item not found"}), 404

    vectors = item.get("vectors") or {}
    objects = []

    # Center / Pore (Class 0)
    pore_poly = vectors.get("pore_poly") or []
    if len(pore_poly) >= 3:
        objects.append({
            "cls": 0,
            "class_name": "center",
            "polygon": pore_poly,
            "source": "model"
        })

    # Stoma Guard Cells (Class 1)
    stoma_poly = vectors.get("stoma_poly") or []
    if len(stoma_poly) >= 3:
        objects.append({
            "cls": 1,
            "class_name": "stoma",
            "polygon": stoma_poly,
            "source": "model"
        })

    # Companion Stomata
    for cs in vectors.get("companion_stomata", []):
        cs_stoma = cs.get("stoma_poly") or []
        if len(cs_stoma) >= 3:
            objects.append({
                "cls": 1,
                "class_name": "stoma",
                "polygon": cs_stoma,
                "source": "model_companion"
            })
        cs_pore = cs.get("pore_poly") or []
        if len(cs_pore) >= 3:
            objects.append({
                "cls": 0,
                "class_name": "center",
                "polygon": cs_pore,
                "source": "model_companion"
            })

    synth = {
        "item_id": item_id,
        "verdict": "stoma" if objects else "no_stoma",
        "objects": objects,
        "labeler": None,
        "saved_at": None,
        "source": "initial_model"
    }
    return jsonify({"ok": True, "item_id": item_id, "label": synth, "objects": objects, "is_saved": False})


@app.post("/api/label/<item_id>")
def api_save_label(item_id):
    """
    Saves in-app ground-truth annotations directly into the dataset without Roboflow!
    Saves JSON metadata and formatted YOLO segmentation polygon .txt file.
    """
    data = request.get_json(force=True)
    m = get_manifest()
    reviewer = getattr(g, "reviewer", None) or data.get("reviewer", "jakub")

    verdict = data.get("verdict", "stoma")  # "stoma", "no_stoma" (hard negative), "skip"
    raw_objects = data.get("objects", [])

    item = next((it for it in m.get("items", []) if it.get("id") == item_id), None)
    if not item:
        return jsonify({"error": "Item not found"}), 404

    crop_w = float(item.get("crop_w", 640))
    crop_h = float(item.get("crop_h", 640))
    stem = Path(item.get("filename", f"{item_id}.jpg")).stem

    valid_objects = []
    yolo_lines = []

    if verdict == "stoma":
        for o in raw_objects:
            poly = o.get("polygon", [])
            if len(poly) < 3:
                continue
            cls_id = int(o.get("cls", 1))  # 0: center/pore, 1: stoma
            cls_name = "center" if cls_id == 0 else "stoma"
            valid_objects.append({
                "cls": cls_id,
                "class_name": cls_name,
                "polygon": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in poly],
                "source": o.get("source", "in_app_studio")
            })

            # YOLO format: <cls> x1 y1 x2 y2 ... xn yn (normalized 0.0 - 1.0)
            coords = []
            for p in poly:
                nx = max(0.0, min(1.0, float(p[0]) / crop_w))
                ny = max(0.0, min(1.0, float(p[1]) / crop_h))
                coords.append(f"{nx:.6f} {ny:.6f}")
            yolo_lines.append(f"{cls_id} " + " ".join(coords))

    elif verdict == "no_stoma":
        # Hard negative crop! Empty label file teaches YOLO to suppress false positives
        valid_objects = []
        yolo_lines = []

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    has_pore = any(o["cls"] == 0 for o in valid_objects)
    is_suitable = bool(data.get("is_suitable_for_ploidy_sizing", has_pore))

    label_doc = {
        "item_id": item_id,
        "stem": stem,
        "filename": item.get("filename"),
        "verdict": verdict,
        "objects": valid_objects,
        "is_suitable_for_ploidy_sizing": is_suitable,
        "is_immature_or_closed": not has_pore,
        "counts_for_density": verdict == "stoma",
        "labeler": reviewer,
        "saved_at": now_iso,
        "seconds": float(data.get("seconds", 0.0)),
        "crop_w": crop_w,
        "crop_h": crop_h,
        "cytotype": item.get("cytotype"),
        "leaf_side": item.get("leaf_side")
    }

    # 1. Save JSON & YOLO TXT locally
    json_path = LABELS_DIR / f"{item_id}.json"
    txt_path = LABELS_DIR / f"{stem}.txt"
    json_path.write_text(json.dumps(label_doc, indent=2))
    txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    # 2. Save to S3
    if s3 is not None:
        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{PREFIX}/ground_truth_labels/{item_id}.json",
                Body=json.dumps(label_doc, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{PREFIX}/ground_truth_labels/{stem}.txt",
                Body=("\n".join(yolo_lines) + ("\n" if yolo_lines else "")).encode("utf-8"),
                ContentType="text/plain"
            )
        except Exception as e:
            print(f"[-] Error writing ground truth label to S3: {e}")

    # 3. Update memory cache
    with lock:
        cache.setdefault("labels", {})[item_id] = label_doc
        # Auto-remove from relabel queue once resolved
        cache.setdefault("relabel", {}).pop(item_id, None)

    return jsonify({
        "ok": True,
        "item_id": item_id,
        "verdict": verdict,
        "objects_count": len(valid_objects),
        "saved_at": now_iso,
        "yolo_lines_count": len(yolo_lines)
    })


@app.get("/api/labels")
def api_get_all_labels():
    """Lists all in-app saved ground-truth labels and summary counts."""
    gt = get_labels()
    stoma_count = sum(len([o for o in doc.get("objects", []) if o.get("cls") == 1]) for doc in gt.values())
    pore_count = sum(len([o for o in doc.get("objects", []) if o.get("cls") == 0]) for doc in gt.values())
    negative_count = sum(1 for doc in gt.values() if doc.get("verdict") == "no_stoma")

    return jsonify({
        "total": len(gt),
        "total_labeled": len(gt),
        "stoma_masks_count": stoma_count,
        "pore_masks_count": pore_count,
        "hard_negatives_count": negative_count,
        "items": list(gt.keys()),
        "labels": gt
    })


@app.post("/api/relabel")
def api_relabel():
    """Enqueues or removes an item from the Relabel Queue."""
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
        if s3 is not None:
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
        if s3 is not None:
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


@app.post("/api/export/stage_training")
def api_stage_training():
    """
    Builds a complete YOLOv8/YOLO11 instance segmentation dataset directly from
    in-app labels + verified high precision detections, without needing Roboflow!
    Uploads dataset zip to S3 Ostatni/Stomata/staged_area/ for automated retraining.
    """
    m = get_manifest()
    dec = get_decisions()
    gt_labels = get_labels()

    items = m.get("items", [])
    if not items:
        return jsonify({"error": "No items available in manifest"}), 400

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"stomata_dataset_{stamp}.zip"
    s3_target_key = f"Ostatni/Stomata/staged_area/{zip_filename}"

    buf = io.BytesIO()
    counts = {"train": 0, "val": 0, "stomata": 0, "pores": 0, "negatives": 0}

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. data.yaml (YOLOv8/11 standard)
        data_yaml = (
            "path: .\n"
            "train: train/images\n"
            "val: val/images\n"
            "nc: 2\n"
            "names:\n"
            "  0: center\n"
            "  1: stoma\n"
        )
        zf.writestr("data.yaml", data_yaml)

        # 2. Stratified train/val split (85% train, 15% val)
        csv_buf = io.StringIO()
        csv_writer = csv.writer(csv_buf)
        csv_writer.writerow(["item_id", "filename", "split", "source", "cytotype", "leaf_side", "n_objects"])

        for it in items:
            item_id = it.get("id")
            fn = it.get("filename", f"{item_id}.jpg")
            stem = Path(fn).stem

            # Determine split deterministically from item_id hash (15% val)
            is_val = (int(hashlib.md5(item_id.encode()).hexdigest(), 16) % 100) < 15
            split_dir = "val" if is_val else "train"

            # Check image bytes
            img_bytes = None
            local_crop = HERE / "static" / "crops" / fn
            if not local_crop.exists():
                local_crop = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "images" / fn
            if local_crop.exists():
                img_bytes = local_crop.read_bytes()
            elif s3 is not None and it.get("crop_s3_key"):
                try:
                    img_bytes = s3.get_object(Bucket=BUCKET, Key=it["crop_s3_key"])["Body"].read()
                except Exception:
                    pass

            if not img_bytes:
                continue

            # Determine labels source:
            # 1. In-app ground truth
            # 2. Local verified TXT file
            # 3. Model vectors if item was evaluated as high_precision
            yolo_lines = []
            source_type = "unknown"

            if item_id in gt_labels:
                source_type = "in_app_ground_truth"
                gt_doc = gt_labels[item_id]
                if gt_doc.get("verdict") == "no_stoma":
                    # Hard negative
                    yolo_lines = []
                    counts["negatives"] += 1
                else:
                    for obj in gt_doc.get("objects", []):
                        cls_id = int(obj.get("cls", 1))
                        poly = obj.get("polygon", [])
                        if len(poly) < 3:
                            continue
                        coords = []
                        for p in poly:
                            nx = max(0.0, min(1.0, float(p[0]) / 640.0))
                            ny = max(0.0, min(1.0, float(p[1]) / 640.0))
                            coords.append(f"{nx:.6f} {ny:.6f}")
                        yolo_lines.append(f"{cls_id} " + " ".join(coords))
                        if cls_id == 1:
                            counts["stomata"] += 1
                        else:
                            counts["pores"] += 1

            elif (LABELS_DIR / f"{stem}.txt").exists():
                source_type = "local_txt"
                raw_lines = (LABELS_DIR / f"{stem}.txt").read_text().splitlines()
                yolo_lines = [l.strip() for l in raw_lines if l.strip()]

            else:
                # Check if item is background control or evaluated high precision
                is_bg = it.get("is_background_control")
                item_dec = dec.get(item_id, {})
                is_high_eval = any(d.get("verdict") == "high_precision" for d in item_dec.values())

                if is_bg:
                    source_type = "background_negative"
                    yolo_lines = []
                    counts["negatives"] += 1
                elif is_high_eval or it.get("stoma_conf", 0) >= 0.65:
                    source_type = "model_verified"
                    # Load pre-annotated dataset txt if exists
                    pre_lbl = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "labels" / f"{stem}.txt"
                    if pre_lbl.exists():
                        yolo_lines = [l.strip() for l in pre_lbl.read_text().splitlines() if l.strip()]
                    else:
                        v = it.get("vectors") or {}
                        # pore
                        if v.get("pore_poly") and len(v["pore_poly"]) >= 3:
                            c_str = [f"{max(0.0, min(1.0, pt[0]/640.0)):.6f} {max(0.0, min(1.0, pt[1]/640.0)):.6f}" for pt in v["pore_poly"]]
                            yolo_lines.append("0 " + " ".join(c_str))
                        # stoma
                        if v.get("stoma_poly") and len(v["stoma_poly"]) >= 3:
                            c_str = [f"{max(0.0, min(1.0, pt[0]/640.0)):.6f} {max(0.0, min(1.0, pt[1]/640.0)):.6f}" for pt in v["stoma_poly"]]
                            yolo_lines.append("1 " + " ".join(c_str))

            # Write into split
            zf.writestr(f"{split_dir}/images/{fn}", img_bytes)
            zf.writestr(f"{split_dir}/labels/{stem}.txt", "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            counts[split_dir] += 1
            csv_writer.writerow([item_id, fn, split_dir, source_type, it.get("cytotype"), it.get("leaf_side"), len(yolo_lines)])

        zf.writestr("dataset_manifest.csv", csv_buf.getvalue())

    zip_bytes = buf.getvalue()

    # Save locally to exports
    local_zip_path = EXPORTS_DIR / zip_filename
    local_zip_path.write_bytes(zip_bytes)
    (EXPORTS_DIR / "latest_stomata_dataset.zip").write_bytes(zip_bytes)

    # Upload to S3 staged area for train_stomata_s3.py
    s3_uploaded = False
    if s3 is not None:
        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=s3_target_key,
                Body=zip_bytes,
                ContentType="application/zip"
            )
            s3_uploaded = True
            print(f"[+] Successfully staged dataset to s3://{BUCKET}/{s3_target_key}")
        except Exception as e:
            print(f"[-] Warning: Failed to upload staged dataset to S3: {e}")

    return jsonify({
        "ok": True,
        "filename": zip_filename,
        "local_path": str(local_zip_path),
        "s3_key": s3_target_key if s3_uploaded else None,
        "s3_staged": s3_uploaded,
        "size_mb": round(len(zip_bytes) / (1024 * 1024), 2),
        "counts": counts
    })


@app.get("/api/export/dataset.zip")
def api_download_dataset():
    """Downloads the latest staged YOLO instance segmentation dataset zip."""
    latest_zip = EXPORTS_DIR / "latest_stomata_dataset.zip"
    if not latest_zip.exists():
        # Trigger build if none exists
        api_stage_training()

    if latest_zip.exists():
        return send_file(
            latest_zip,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"stomata_yolo_dataset_{time.strftime('%Y%m%d')}.zip"
        )
    return Response("No dataset generated yet.", status=404)


@app.get("/api/export/roboflow_relabel.zip")
def export_roboflow_relabel():
    """Legacy export endpoint maintained for backwards compatibility."""
    m = get_manifest()
    rq = get_relabel_queue()
    if not rq:
        return Response("No crops flagged for relabeling yet.", status=400, mimetype="text/plain")

    items_by_id = {it["id"]: it for it in m.get("items", [])}
    slides_by_stem = {sl["stem"]: sl for sl in m.get("slides", [])}

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        data_yaml = "names:\n- center\n- stoma\nnc: 2\n"
        zf.writestr("data.yaml", data_yaml)

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

                img_bytes = None
                local_crop = HERE / "static" / "crops" / fn
                if not local_crop.exists():
                    local_crop = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "images" / fn
                if local_crop.exists():
                    img_bytes = local_crop.read_bytes()
                elif s3 is not None and it.get("crop_s3_key"):
                    try:
                        img_bytes = s3.get_object(Bucket=BUCKET, Key=it["crop_s3_key"])["Body"].read()
                    except Exception:
                        pass

                if img_bytes:
                    zf.writestr(f"train/images/{fn}", img_bytes)

                lbl_lines = []
                local_lbl = LABELS_DIR / f"{stem}.txt"
                if not local_lbl.exists():
                    local_lbl = HERE.parent.parent / "roboflow_preannotated_dataset_all_300" / "labels" / f"{stem}.txt"
                if local_lbl.exists():
                    lbl_lines = local_lbl.read_text().splitlines()

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
                elif s3 is not None and sl.get("overview_s3_key"):
                    try:
                        img_bytes = s3.get_object(Bucket=BUCKET, Key=sl["overview_s3_key"])["Body"].read()
                        zf.writestr(f"train/images/{fn}", img_bytes)
                    except Exception:
                        pass

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
    print(f"🚀 Stomata Detection QC & Label Studio serving on http://0.0.0.0:{port}")
    serve(app, host="0.0.0.0", port=port, threads=8)
