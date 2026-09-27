#!/bin/bash
#SBATCH -J final_chain
#SBATCH -p debug6
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:29:00
#SBATCH -o slurm_logs/chain-%j.out
#SBATCH -e slurm_logs/chain-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

TASK_ID="${FINAL_TASK_ID:?FINAL_TASK_ID is required}"
CHAIN_ID="${FINAL_CHAIN_ID:?FINAL_CHAIN_ID is required}"
STRIDE=4
if [ "$CHAIN_ID" -lt 1 ] || [ "$CHAIN_ID" -gt 4 ]; then
    echo "ERROR: FINAL_CHAIN_ID must be 1-4" >&2
    exit 2
fi
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 48 ]; then
    echo "ERROR: FINAL_TASK_ID must be 1-48" >&2
    exit 3
fi

module purge
module load Intel_compiler/23.0.0
module load MPI/mpich/4.0.2-mpi-x-icc23.0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

SISSO_EXE=/fs2/home/chenyifei/JCAT_SISSO/sisso_allD_hpc/bin/SISSO_allD_final
test -x "$SISSO_EXE"
test -f results/full_data_regression.ok

echo "CHAIN_ID=$CHAIN_ID"
echo "TASK_ID=$TASK_ID"
echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
python3 -u scripts/run_outer_task.py \
    --task-id "$TASK_ID" \
    --sisso-exe "$SISSO_EXE" \
    --mpi-tasks "${SLURM_NTASKS:-8}"

CASE_REL="$(awk -F, -v id="$TASK_ID" 'NR>1 && $1==id {print $2}' cases_manifest.csv)"
test -n "$CASE_REL"
test -f "$CASE_REL/outer_complete.ok"

NEXT_TASK=$((TASK_ID + STRIDE))
if [ "$NEXT_TASK" -le 48 ]; then
    NEXT_JOB=""
    for attempt in $(seq 1 20); do
        if NEXT_JOB=$(sbatch --parsable \
            --chdir="$ROOT" \
            --export="ALL,FINAL_TASK_ID=${NEXT_TASK},FINAL_CHAIN_ID=${CHAIN_ID}" \
            "$ROOT/03_run_chain_debug.slurm" 2>/dev/null); then
            break
        fi
        echo "RESUBMIT_RETRY=$attempt NEXT_TASK=$NEXT_TASK"
        sleep 30
    done
    if [ -z "$NEXT_JOB" ]; then
        echo "ERROR: could not submit next chain task $NEXT_TASK after 20 attempts" >&2
        exit 4
    fi
    echo "NEXT_TASK=$NEXT_TASK"
    echo "NEXT_JOB=$NEXT_JOB"
else
    echo "CHAIN_COMPLETE=$CHAIN_ID"
fi
echo "END=$(date --iso-8601=seconds)"

