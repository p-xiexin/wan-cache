#!/usr/bin/env bash
#SBATCH --job-name=wan-cache-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=8

cd /path/to/EasyCache
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate easycache

python -m eval.pipeline \
  parallel.num_processes=auto \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B \
  paths.model_artifact=/path/to/EasyCache/eval/artifacts/model/model.pth \
  paths.temporal_artifact=/path/to/EasyCache/eval/artifacts/temporal/model.pth \
  paths.output_root=/path/to/EasyCache/eval/outputs/experiment
