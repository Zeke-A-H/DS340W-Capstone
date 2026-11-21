#!/bin/bash
#SBATCH --job-name=img2cad-original
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/original-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec bash tools/run-linux.sh tools/run_original_baseline.py "$@"
