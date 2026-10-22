#!/bin/bash
#SBATCH --job-name=img2cad-test
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "Img2CAD GPU session on $(hostname), job ${SLURM_JOB_ID}"
nvidia-smi
echo "Connect from the login node:"
echo "srun --jobid=${SLURM_JOB_ID} --overlap --pty bash -l"
sleep 3600
