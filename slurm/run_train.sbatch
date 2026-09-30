#!/usr/bin/env bash
#SBATCH --job-name=wan-cache-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

set -euo pipefail

cd "${EASYCACHE_ROOT:-/path/to/EasyCache}"
source "${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
conda activate "${WAN_CONDA_ENV:-easycache}"

python -m eval.train --config-name="${TRAIN_CONFIG:-train_predictor}" \
  data.data_dir="${TRAIN_DATA_DIR:-/path/to/raw_wan_trajectories}" \
  "$@"
