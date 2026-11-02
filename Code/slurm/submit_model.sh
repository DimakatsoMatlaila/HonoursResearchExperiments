#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config.sh"
MODEL="${1:?usage: submit_model.sh <model> [mixed|https|both] [\"internal external master\"]}"
VAR="${2:-both}"
DS="${3:-$DATASET_NAMES}"
[ -f "$PIPE/models/$MODEL/train.py" ] || { echo "unknown model $MODEL (have: $MODELS)"; exit 1; }
[ "$VAR" = "both" ] && VAR="$VARIANTS"

SMOKE="${SMOKE:-0}"
TIME="$TRAIN_TIME"; PART="$TRAIN_PARTITION"
[ "$SMOKE" = "1" ] && { TIME="01:00:00"; PART="bigbatch"; }

STAGE=train; [ "$SMOKE" = "1" ] && STAGE=smoke
N=$(wc -w <<< "$DS")
[ "$N" -gt "$TRAIN_NODES" ] && N="$TRAIN_NODES"

mkdir -p "$PIPE/logs/train"
cd "$PIPE/logs/train"
export MODEL DATASETS_TO_RUN="$DS" SMOKE EXTRA_ARGS="${EXTRA_ARGS:-}"

for V in $VAR; do
    export VARIANT="$V"
    id=$(sbatch --parsable --job-name="c2_${MODEL}_${V}" --partition="$PART" --nodes="$N" --ntasks="$N" \
         --ntasks-per-node=1 --time="$TIME" --exclude="$EXCLUDE_NODES" --export=ALL \
         "$PIPE/slurm/train_model.sbatch")
    log_job "$STAGE" "$MODEL/$V [$DS]" "$id"
    echo "${GRN}$MODEL / $V${RST}  job $id  ($N node(s), $PART, $TIME, not on $EXCLUDE_NODES)"
done
