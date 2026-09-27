#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p slurm_logs results
bash 00_check_project.sh

submit_retry() {
    local output=""
    for attempt in $(seq 1 20); do
        if output=$(sbatch --parsable "$@" 2>&1); then
            printf '%s\n' "$output"
            return 0
        fi
        echo "SBATCH_RETRY=$attempt: $output" >&2
        sleep 30
    done
    echo "ERROR: sbatch failed after 20 attempts" >&2
    return 1
}

GATE_JOB=$(submit_retry --chdir="$ROOT" 01_full_data_gate_debug.slurm)
SMOKE_JOB=$(submit_retry --chdir="$ROOT" --dependency="afterok:${GATE_JOB}" 02_smoke_task1_debug.slurm)

{
    echo "GATE_JOB=$GATE_JOB"
    echo "SMOKE_JOB=$SMOKE_JOB"
} | tee submitted_jobs.txt

for CHAIN_ID in 1 2 3 4; do
    JOB_ID=$(submit_retry \
        --chdir="$ROOT" \
        --dependency="afterok:${SMOKE_JOB}" \
        --export="ALL,FINAL_TASK_ID=${CHAIN_ID},FINAL_CHAIN_ID=${CHAIN_ID}" \
        03_run_chain_debug.slurm)
    echo "CHAIN_${CHAIN_ID}_JOB=$JOB_ID" | tee -a submitted_jobs.txt
done

echo "SUBMIT_OK: full-data gate, one smoke fold, and four resumable chains are queued"
squeue -u "$USER" -o "%.20i %.14j %.2t %.10M %R"
