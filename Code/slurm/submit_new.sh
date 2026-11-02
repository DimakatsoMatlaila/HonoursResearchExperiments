#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config.sh"
export BUNDLE_PARTITION="${NEW_PARTITION:-stampede}" BUNDLE_NODES=1 BUNDLE_TIME=3-00:00:00
export LONG_HOURS=68 SHORT_HOURS=68
export DATASETS_TO_RUN="master"
for m in gwo_rf_nested dqn_gwo_rf_nested; do
    for v in mixed https; do
        MODELS_TO_RUN="$m" VARIANTS_TO_RUN="$v" bash "$PIPE/slurm/submit_bundle.sh"
    done
done
export DATASETS_TO_RUN="master external internal" VARIANTS_TO_RUN="mixed https"
MODELS_TO_RUN="xgb mlp_erm mlp_vrex mlp_gdro" bash "$PIPE/slurm/submit_bundle.sh"
