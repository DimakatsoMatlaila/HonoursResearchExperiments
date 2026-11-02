#!/bin/bash
source "$(dirname "$0")/../config.sh"
for m in $MODELS; do
    SMOKE=1 bash "$PIPE/slurm/submit_model.sh" "$m" mixed internal
done
