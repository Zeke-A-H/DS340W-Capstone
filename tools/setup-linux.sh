#!/bin/bash
# Run from the project root inside a Slurm allocation.
set -euo pipefail
module load miniforge/25.3.0-3
conda activate "$PWD/data/env-linux"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR="$PWD/data/.cache/torch_extensions_linux"
export MAX_JOBS=4
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
python -m pip install 'setuptools==80.9.0' wheel ninja
# Install PyTorch before packages whose builds import it.
python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-build-isolation -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 --find-links https://data.pyg.org/whl/torch-2.5.0+cu118.html
# The PyPI apex package in GMFlow's list is unrelated to NVIDIA Apex.
# NVIDIA Apex is optional for the inference path; do not install PyPI apex.
python - <<'PY'
from pathlib import Path
lines = Path('GMFlow/requirements.txt').read_text().splitlines()
Path('data/requirements-gmflow-linux.txt').write_text('\n'.join(line for line in lines if not line.startswith('apex==')) + '\n')
PY
python -m pip install --no-build-isolation -c requirements.txt -r data/requirements-gmflow-linux.txt
python -m pip install -c requirements.txt flow-matching git+https://github.com/openai/CLIP.git
python -m pip check
python -c "import torch; from OCC.Core.gp import gp_Pnt; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python TrAssembler/eval.py --help
