#!/usr/bin/env bash
#SBATCH --job-name=wan-cache-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8

cd /path/to/EasyCache
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate easycache

python -m eval.train \
  data.data_dir=/path/to/trajectories
