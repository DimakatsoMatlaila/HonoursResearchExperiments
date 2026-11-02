#!/bin/bash
#SBATCH --job-name=c2_clean
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --exclude=mscluster128,mscluster112
#SBATCH --output=%x-%A_%a.out
source "$PIPE/config.sh"
rel=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[ -n "$rel" ] || { echo "no manifest line ${SLURM_ARRAY_TASK_ID}"; exit 1; }
bash "$PIPE/data_pipeline/clean_one.sh" "$rel"
