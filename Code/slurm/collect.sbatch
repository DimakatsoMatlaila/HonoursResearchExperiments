#!/bin/bash
#SBATCH --job-name=c2_collect
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --exclude=mscluster128,mscluster112
#SBATCH --output=%x-%j.out
source "$PIPE/config.sh"
"$PY" "$PIPE/common/collect_results.py" "$PIPE"
