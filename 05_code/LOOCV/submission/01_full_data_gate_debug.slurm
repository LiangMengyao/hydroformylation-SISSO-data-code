#!/bin/bash
#SBATCH -J final_gate
#SBATCH -p debug6
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o slurm_logs/full-gate-%j.out
#SBATCH -e slurm_logs/full-gate-%j.err

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
sha256sum "$SISSO_EXE" > results/SISSO_allD_final.sha256

echo "HOST=$(hostname)"
echo "START=$(date --iso-8601=seconds)"
python3 -u scripts/run_full_data.py --sisso-exe "$SISSO_EXE" --mpi-tasks "${SLURM_NTASKS:-8}"
echo "END=$(date --iso-8601=seconds)"

