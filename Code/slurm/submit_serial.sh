#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config.sh"
PART="${SERIAL_PARTITION:-biggpu}"
TIME="${SERIAL_TIME:-3-00:00:00}"
mkdir -p "$PIPE/logs/train"
cd "$PIPE/logs/train"
export MODELS_TO_RUN VARIANTS_TO_RUN DATASETS_TO_RUN WAIT_FOR_JOB EXTRA_ARGS="${EXTRA_ARGS:-}"
id=$(sbatch --parsable --partition="$PART" --nodes=1 --ntasks=1 --time="$TIME" \
     --exclude="$EXCLUDE_NODES" ${GRES:+--gres=$GRES} --export=ALL "$PIPE/slurm/train_serial.sbatch")
log_job serial "1 node ${PART}, ${MODELS_TO_RUN:-all models}" "$id"
echo "${GRN}serial${RST} job $id  (1 node, $PART, $TIME)"
