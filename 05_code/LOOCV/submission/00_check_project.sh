#!/bin/bash
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
python3 scripts/static_check.py
SISSO_EXE=/fs2/home/chenyifei/JCAT_SISSO/sisso_allD_hpc/bin/SISSO_allD_final
test -x "$SISSO_EXE"
echo "SISSO_ALLD_OK: $SISSO_EXE"

