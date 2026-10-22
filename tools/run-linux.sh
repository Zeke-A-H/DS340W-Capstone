#!/bin/bash
# Run from a GPU allocation. Paths work from any current directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load miniforge/25.3.0-3
conda activate "$PWD/data/env-linux"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR="$PWD/data/.cache/torch_extensions_linux"
export MAX_JOBS=4
exec python "$@"
