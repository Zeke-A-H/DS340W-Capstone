#!/bin/bash
# Run from a GPU allocation. Paths work from any current directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
module load miniforge/25.3.0-3
module load gcc/9.1.0
conda activate "$PWD/data/env-linux"
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR="$PWD/data/.cache/torch_extensions_linux"
export HF_HOME="${HF_HOME:-$PWD/data/.cache/huggingface}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
exec python "$@"
