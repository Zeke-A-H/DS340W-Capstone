#!/bin/bash
#SBATCH --job-name=img2cad-scenes
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scenes-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$PWD/data/.cache/matplotlib"
exec "$PWD/data/env-linux/bin/python" SceneAssembler/assemble.py "$@"
