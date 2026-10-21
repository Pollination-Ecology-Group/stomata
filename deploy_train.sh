#!/bin/bash
set -e

NAMESPACE="stenc-ns"
JOB_NAME="stomata-train-job"
CONFIGMAP_NAME="stomata-train-script"

echo "=== Deploying Stomata Retraining Job to CESNET Kubernetes ==="

# 1. Update/create ConfigMap with the training script
echo "[*] Updating ConfigMap: ${CONFIGMAP_NAME} in namespace ${NAMESPACE}..."
kubectl create configmap "${CONFIGMAP_NAME}" \
  --from-file=train_stomata_s3.py=src/train_stomata_s3.py \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Remove existing job if any
echo "[*] Cleaning up old job if present..."
kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found

# 3. Apply the Job manifest
echo "[*] Launching training job..."
kubectl apply -f k8s/stomata-train-job.yaml

echo "=== Job submitted successfully! ==="
echo "To monitor status:"
echo "  kubectl get job ${JOB_NAME} -n ${NAMESPACE}"
echo "To follow training logs:"
echo "  kubectl logs -f job/${JOB_NAME} -n ${NAMESPACE}"
