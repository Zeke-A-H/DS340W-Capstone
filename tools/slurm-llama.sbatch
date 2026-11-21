#!/bin/bash
#SBATCH --job-name=img2cad-llama
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/llama-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [ "$#" -eq 0 ]; then
    echo 'Usage: sbatch tools/slurm-llama.sbatch LlamaFT/finetune.py [arguments]' >&2
    exit 2
fi
exec bash tools/run-linux.sh "$@"
