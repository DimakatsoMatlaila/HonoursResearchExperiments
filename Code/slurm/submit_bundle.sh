#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config.sh"
NODES="${BUNDLE_NODES:-6}"
PART="${BUNDLE_PARTITION:-bigbatch}"
SMOKE="${SMOKE:-0}"
TIME="${BUNDLE_TIME:-3-00:00:00}"
[ "$SMOKE" = "1" ] && TIME="02:00:00"

mkdir -p "$PIPE/logs/train"
cd "$PIPE/logs/train"
export SMOKE MODELS_TO_RUN VARIANTS_TO_RUN DATASETS_TO_RUN EXTRA_ARGS="${EXTRA_ARGS:-}"
id=$(sbatch --parsable --partition="$PART" --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
     --time="$TIME" --exclude="$EXCLUDE_NODES" --export=ALL ${DEPEND:+--dependency=$DEPEND --kill-on-invalid-dep=yes} "$PIPE/slurm/train_bundle.sbatch")
log_job bundle "$NODES nodes, ${MODELS_TO_RUN:-all models}" "$id"
echo "${GRN}bundle${RST} job $id  ($NODES nodes, $PART, $TIME, not on $EXCLUDE_NODES)"
echo "${DIM}resubmit the same command after a timeout; finished runs are skipped and folds resume${RST}"
