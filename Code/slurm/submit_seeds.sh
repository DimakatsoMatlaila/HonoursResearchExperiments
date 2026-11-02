#!/bin/bash
# SEED=1337 TIER=cheap|search|nested bash slurm/submit_seeds.sh
set -euo pipefail
source "$(dirname "$0")/../config.sh"
: "${SEED:?set SEED, e.g. SEED=1337}"
TIER="${TIER:-cheap}"
export C2_SEED="$SEED" RESDIR="results_seed${SEED}"
export BUNDLE_PARTITION="${BUNDLE_PARTITION:-bigbatch}" BUNDLE_NODES="${BUNDLE_NODES:-6}" BUNDLE_TIME="${BUNDLE_TIME:-3-00:00:00}"
export DATASETS_TO_RUN="master external internal" VARIANTS_TO_RUN="mixed https"
export LONG_HOURS=68 SHORT_HOURS=12
case "$TIER" in
  cheap)  export MODELS_TO_RUN="rf_baseline hist_gb xgb mlp_erm mlp_vrex mlp_gdro lsh_rf isolation_forest" ;;
  search) export MODELS_TO_RUN="gwo_rf dqn_gwo_rf" ;;
  nested) export MODELS_TO_RUN="gwo_rf_nested dqn_gwo_rf_nested" SHORT_HOURS=34 ;;
  *) echo "TIER must be cheap, search or nested"; exit 1 ;;
esac
echo "seed $SEED, tier $TIER, results in models/<model>/$RESDIR/"
bash "$PIPE/slurm/submit_bundle.sh"
