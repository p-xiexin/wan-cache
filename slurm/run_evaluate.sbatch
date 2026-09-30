#!/usr/bin/env bash
#SBATCH --job-name=wan-cache-metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

set -euo pipefail

cd "${EASYCACHE_ROOT:-/path/to/EasyCache}"
source "${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
conda activate "${WAN_CONDA_ENV:-easycache}"

python -m eval.evaluate \
  --config "${QUALITY_CONFIG:-eval/conf/quality_predictor.yaml}" \
  "$@"
