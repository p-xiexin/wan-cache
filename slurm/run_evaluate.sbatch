#!/usr/bin/env bash
#SBATCH --job-name=wan-cache-metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

set -e

cd /path/to/EasyCache
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate easycache

python -m eval.evaluate \
  --config eval/conf/quality.yaml
