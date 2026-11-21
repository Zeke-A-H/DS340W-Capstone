#!/bin/bash
#SBATCH --job-name=img2cad-trassembler
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/trassembler-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [ "$#" -eq 0 ]; then
    echo 'Usage: sbatch tools/slurm-trassembler.sbatch TrAssembler/train.py [Hydra overrides]' >&2
    exit 2
fi
exec bash tools/run-linux.sh "$@"
