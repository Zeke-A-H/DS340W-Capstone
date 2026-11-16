#!/bin/bash
#SBATCH --job-name=cad-scenes-v2
#SBATCH --account=open
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scenes-v2-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load blender/4.5.1
if [ ! -f data/scene_dataset_v2/_build/recipes/manifest.json ]; then
    data/env-linux/bin/python tools/create_scene_dataset.py --resume
fi
blender -b --python-exit-code 1 --python tools/render_scene_pilot_blender.py -- \
    --input data/scene_dataset_v2/_build/recipes --output data/scene_dataset_v2/_build/blender \
    --background flat-gray --device CUDA --threads "${SLURM_CPUS_PER_TASK}"
# Resume rendering and any incomplete packaging after a job time limit.
data/env-linux/bin/python tools/package_scene_dataset.py \
    --input data/scene_dataset_v2/_build/blender --output data/scene_dataset_v2 --resume --simple-layout
data/env-linux/bin/python tools/prepare_scene_crops.py --dataset data/scene_dataset_v2
data/env-linux/bin/python tools/validate_scene_dataset.py --dataset data/scene_dataset_v2
