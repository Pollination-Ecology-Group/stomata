#!/bin/bash
# Deploy the CPU report job that waits for the train/inference job, renders
# short_report.qmd to PDF and e-mails it.
#   bash deploy_report.sh                 # real run (RUN_ID from the manifest)
#   bash deploy_report.sh test-report     # test: separate job name, given RUN_ID, no e-mail
set -e
NAMESPACE="stenc-ns"
CONFIGMAP_NAME="stomata-report-files"

kubectl create configmap "${CONFIGMAP_NAME}" \
  --from-file=short_report.qmd=short_report.qmd \
  --from-file=report_fetch.py=src/report_fetch.py \
  --from-file=report_email.py=src/report_email.py \
  -n "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

if [ -n "$1" ]; then
  JOB_NAME="stomata-report-test"
  kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found
  sed -e "s/name: stomata-report-job/name: ${JOB_NAME}/" \
      -e "s/value: \"v8-20261022\"/value: \"$1\"/" \
      k8s/stomata-report-job.yaml \
    | python3 -c "import sys; s=sys.stdin.read(); s=s.replace('- name: DRY_RUN\n            value: \"false\"','- name: DRY_RUN\n            value: \"true\"').replace('value: \"300\"','value: \"10\"'); print(s)" \
    | kubectl apply -f -
else
  JOB_NAME="stomata-report-job"
  kubectl delete job "${JOB_NAME}" -n "${NAMESPACE}" --ignore-not-found
  kubectl apply -f k8s/stomata-report-job.yaml
fi
echo "Submitted ${JOB_NAME}. Logs: kubectl logs -f job/${JOB_NAME} -n ${NAMESPACE} --all-containers"
