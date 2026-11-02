#!/bin/bash
source "$(dirname "$0")/../config.sh"
for m in $MODELS; do
    bash "$PIPE/slurm/submit_model.sh" "$m" "${1:-both}" "${2:-$DATASET_NAMES}"
done
