#!/bin/bash
PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIPE
export PY="${PY:-$HOME/anaconda3/envs/pcapenv/bin/python}"

export RAW="${RAW:-/datasets/dmatlaila/pcaps_raw}"
export WORK="${WORK:-/datasets/dmatlaila/c2_work}"
export CLEAN="$WORK/pcaps_clean"
export LABELED="$WORK/labeled"
export DATASETS="$WORK/datasets"
export C2_IPS_FILE="$PIPE/c2_ips.txt"
export MANIFEST="$PIPE/manifests/pcap_manifest.txt"
export JOBLOG="$PIPE/logs/jobs.tsv"
export FAILLOG="$PIPE/logs/failed_items.tsv"

# set to reuse old labelled CSVs (only for profiles whose C2 IP is unchanged)
export REUSE_LABELED_FROM="${REUSE_LABELED_FROM:-}"

export EXCLUDE_NODES="${EXCLUDE_NODES:-mscluster128,mscluster112}"
export MAX_PARALLEL="${MAX_PARALLEL:-6}"
export CLEAN_PARTITION="bigbatch"; export CLEAN_TIME="3-00:00:00"
export LABEL_PARTITION="bigbatch"; export LABEL_TIME="3-00:00:00"
export BUILD_PARTITION="${BUILD_PARTITION:-bigbatch}"; export BUILD_TIME="${BUILD_TIME:-1-00:00:00}"
export TRAIN_PARTITION="${TRAIN_PARTITION:-bigbatch}"; export TRAIN_TIME="${TRAIN_TIME:-3-00:00:00}"
export TRAIN_NODES="${TRAIN_NODES:-3}"
export TRAIN_MAX_HOURS="${TRAIN_MAX_HOURS:-68}"

export MODELS="dqn_gwo_rf gwo_rf rf_baseline isolation_forest lsh_rf hist_gb xgb mlp_erm mlp_vrex mlp_gdro gwo_rf_nested dqn_gwo_rf_nested"
export DATASET_NAMES="internal external master"
export VARIANTS="mixed https"

if [ -t 1 ]; then
    RED=$'\e[38;5;167m'; GRN=$'\e[38;5;108m'; YEL=$'\e[38;5;179m'; BLU=$'\e[38;5;74m'; DIM=$'\e[38;5;245m'; RST=$'\e[0m'
else
    RED=""; GRN=""; YEL=""; BLU=""; DIM=""; RST=""
fi

log_job() {
    mkdir -p "$PIPE/logs"
    printf '%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$1" "$2" "$3" >> "$JOBLOG"
}

mark_failed() {
    mkdir -p "$PIPE/logs"
    printf '%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$1" "$2" "${SLURM_JOB_ID:-local}" >> "$FAILLOG"
}
