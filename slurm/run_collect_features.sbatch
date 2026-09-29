#!/usr/bin/env bash
#SBATCH --job-name=wan-feature-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=8

cd /path/to/EasyCache
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate easycache

python -m eval.collect_features \
  parallel.num_processes=auto \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B \
  paths.output_root=/path/to/EasyCache/eval/data/cache_features \
  prompts.file=/path/to/prompts.txt
