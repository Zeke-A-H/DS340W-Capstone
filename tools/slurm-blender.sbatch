#!/bin/bash
#SBATCH --job-name=cad-blender
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/blender-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load blender/4.5.1
blender -b --python-exit-code 1 --python tools/render_scene_pilot_blender.py -- --device CUDA --threads "${SLURM_CPUS_PER_TASK}" "$@"
