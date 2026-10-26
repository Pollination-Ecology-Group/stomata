#!/usr/bin/env bash
# Deploy or update the Stomata Measurement Precision Review App on CESNET Kubernetes.
# Usage:
#   bash deploy_stomata_review.sh
set -euo pipefail
cd "$(dirname "$0")"

NS="stenc-ns"
CRED="src/review_app/.credentials"

if [[ "${1:-}" == "--reset-passwords" ]]; then
  shift; users=("$@"); [[ ${#users[@]} -eq 0 ]] && users=(jakub eliska)
  : > "$CRED"; chmod 600 "$CRED"
  for u in "${users[@]}"; do echo "$u  $(openssl rand -base64 18 | tr -dc 'A-Za-z0-9' | head -c 14)" >> "$CRED"; done
  echo "🔑 New credentials:"; cat "$CRED"
  python3 - "$CRED" > /tmp/stomata_users.json <<'PY'
import hashlib, json, os, sys
users = {}
for line in open(sys.argv[1]):
    if line.strip():
        u, pw = line.split()
        salt = os.urandom(16); it = 200_000
        users[u.lower()] = f"pbkdf2${it}${salt.hex()}${hashlib.pbkdf2_hmac('sha256', pw.encode(), salt, it).hex()}"
print(json.dumps(users))
PY
  kubectl create secret generic viability-review-users -n "$NS" --from-file=users.json=/tmp/stomata_users.json \
    --dry-run=client -o yaml | kubectl apply -f -
  rm -f /tmp/stomata_users.json
fi

# Sync .credentials from Pollen_viability if missing
if [[ ! -s "$CRED" && -s ../Pollen_viability/review_app/.credentials ]]; then
  cp ../Pollen_viability/review_app/.credentials "$CRED"
  chmod 600 "$CRED"
fi

echo "🌿 Deploying Stomata Measurement Precision Review App to CESNET Kubernetes ($NS)..."

# 1. Create or update ConfigMap with application source code
echo "📦 Updating ConfigMap stomata-review-src..."
kubectl create configmap stomata-review-src -n "$NS" \
  --from-file=app.py=src/review_app/app.py \
  --from-file=index.html=src/review_app/index.html \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Apply deployment, service, and ingress
echo "🚀 Applying Kubernetes manifests (deployment, service, ingress)..."
kubectl apply -f k8s/stomata-review.yaml

# 3. Rollout restart
echo "🔄 Restarting stomata-review deployment to load latest code..."
kubectl rollout restart deployment/stomata-review -n "$NS"

echo "⏳ Waiting for Stomata Review pod to become ready..."
kubectl rollout status deployment/stomata-review -n "$NS" --timeout=180s

echo "============================================================"
echo "✅ Stomata Measurement Precision App is LIVE on CESNET!"
echo "🌐 URL: https://stomata-review-stenc.dyn.cloud.e-infra.cz"
echo "============================================================"
