#!/bin/bash
#SBATCH -J final_smoke
#SBATCH -p debug6
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o slurm_logs/smoke-%j.out
#SBATCH -e slurm_logs/smoke-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

module purge
module load Intel_compiler/23.0.0
module load MPI/mpich/4.0.2-mpi-x-icc23.0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -s unlimited

SISSO_EXE=/fs2/home/chenyifei/JCAT_SISSO/sisso_allD_hpc/bin/SISSO_allD_final
test -x "$SISSO_EXE"
test -f results/full_data_regression.ok

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
python3 -u scripts/run_outer_task.py --task-id 1 --sisso-exe "$SISSO_EXE" --mpi-tasks "${SLURM_NTASKS:-8}"
test -f cases/001_M_LP1/outer_complete.ok
echo "SMOKE_TASK1: PASS"
echo "END=$(date --iso-8601=seconds)"

