#!/bin/bash
# bash slurm/submit_analysis.sh dukes|cross   (one batch node each, nothing on the login node)
set -euo pipefail
source "$(dirname "$0")/../config.sh"
mkdir -p "$PIPE/logs/train" "$PIPE/analysis"
cd "$PIPE/logs/train"
SEEDS="${SEEDS:-42 1337 2024}"
case "${1:-}" in
  dukes)
    cmd="for s in $SEEDS; do for v in mixed https; do C2_SEED=\$s $PY $PIPE/common/dukes_missed.py --data $DATASETS/master_dataset.csv --variant \$v --out-dir $PIPE/analysis/dukes_\${v}_seed\$s; done; done" ;;
  cross)
    cmd="for s in $SEEDS; do C2_SEED=\$s $PY $PIPE/common/cross_protocol.py --data $DATASETS/master_dataset.csv --out-dir $PIPE/analysis/cross_protocol_seed\$s; done" ;;
  *) echo "usage: submit_analysis.sh dukes|cross"; exit 1 ;;
esac
id=$(sbatch --parsable --partition="${ANALYSIS_PARTITION:-batch}" --nodes=1 --exclusive \
     --time=1-00:00:00 --exclude="$EXCLUDE_NODES" --job-name="c2_$1" --output="%x-%j.out" --wrap="$cmd")
log_job analysis "$1" "$id"
echo "analysis job $id ($1), output in $PIPE/analysis/"
