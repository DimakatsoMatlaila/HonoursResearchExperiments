#!/bin/bash
#SBATCH --job-name=c2_build
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --exclude=mscluster128,mscluster112
#SBATCH --output=%x-%j.out
source "$PIPE/config.sh"
"$PY" "$PIPE/common/build_datasets.py" "$LABELED" "$DATASETS"
