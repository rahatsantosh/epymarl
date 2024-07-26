#!/bin/bash
#
#SBATCH --job-name=epymarl_batch
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=160:00:00
#SBATCH --mem-per-cpu=10000M
#
#SBATCH --array=0-45%20
#SBATCH --requeue
#SBATCH --nice

source ~/anaconda3/bin/activate
conda activate am

python src/search.py run --config epymarl_search_config.yaml --seeds=5 single $SLURM_ARRAY_TASK_ID