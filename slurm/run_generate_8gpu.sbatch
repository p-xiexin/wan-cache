#!/usr/bin/env bash
#SBATCH --job-name=wan-cache-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=8

set -euo pipefail

cd "${EASYCACHE_ROOT:-/path/to/EasyCache}"
source "${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
conda activate "${WAN_CONDA_ENV:-easycache}"

python -m eval.pipeline --config-name="${GENERATE_CONFIG:-sweep_predictor}" \
  parallel.num_processes=auto \
  paths.checkpoint_dir="${WAN_CKPT_DIR:-/path/to/Wan2.2-TI2V-5B}" \
  paths.predictor_artifact="${PREDICTOR_ARTIFACT:-eval/artifacts/polynomial/model.pth}" \
  "$@"
