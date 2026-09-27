#!/bin/bash
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
python3 scripts/static_check.py
python3 scripts/collect_results.py

