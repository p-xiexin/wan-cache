#!/usr/bin/env bash
#SBATCH --job-name=wan-raw-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32

cd /path/to/EasyCache
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate easycache

python -m eval.collect_raw_trajectories \
  paths.checkpoint_dir=/path/to/Wan2.2-TI2V-5B \
  paths.output_root=/path/to/raw_wan_trajectories \
  data.prompt_file=/path/to/prompts.txt \
  parallel.num_processes=8
