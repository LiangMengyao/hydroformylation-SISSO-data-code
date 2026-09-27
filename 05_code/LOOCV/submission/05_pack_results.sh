#!/bin/bash
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
test -f results/collection.ok
ARCHIVE=LOOCV_results.tar.gz
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" \
    --exclude='feature_space' \
    --exclude='SIS_subspaces' \
    --exclude='__pycache__' \
    input reference full_data cases truth results scripts \
    cases_manifest.csv source_manifest.csv provenance.json \
    README_FINAL_LOOCV.md \
    00_check_project.sh 01_full_data_gate_debug.slurm \
    02_smoke_task1_debug.slurm 03_run_chain_debug.slurm \
    04_collect_results.sh 05_pack_results.sh 06_submit_all.sh
echo "PACK_OK: $ROOT/$ARCHIVE"
