#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../config.sh"
STAGES="${STAGES:-clean label build}"

mkdir -p "$CLEAN" "$LABELED" "$DATASETS" "$PIPE/logs/data"
bash "$PIPE/data_pipeline/make_manifest.sh"
N=$(wc -l < "$MANIFEST")

while read -r rel; do
    mkdir -p "$CLEAN/$(dirname "$rel")" "$LABELED/$(basename "$(dirname "$rel")")" 2>/dev/null || true
done < "$MANIFEST"

cd "$PIPE/logs/data"
export PIPE
COMMON=(--export=ALL --exclude="$EXCLUDE_NODES")
cj=""; lj=""

if [[ " $STAGES " == *" clean "* ]]; then
    cj=$(sbatch --parsable "${COMMON[@]}" --partition="$CLEAN_PARTITION" --time="$CLEAN_TIME" \
         --array=1-"$N"%"$MAX_PARALLEL" "$PIPE/data_pipeline/clean_array.sbatch")
    log_job clean "array 1-$N" "$cj"
    echo "${GRN}cleaning${RST}   job $cj  (1-$N, $MAX_PARALLEL at a time, not on $EXCLUDE_NODES)"
fi

if [[ " $STAGES " == *" label "* ]]; then
    dep=()
    [ -n "$cj" ] && dep=(--dependency=aftercorr:"$cj" --kill-on-invalid-dep=yes)
    lj=$(sbatch --parsable "${COMMON[@]}" --partition="$LABEL_PARTITION" --time="$LABEL_TIME" \
         --array=1-"$N"%"$MAX_PARALLEL" "${dep[@]}" "$PIPE/data_pipeline/label_array.sbatch")
    log_job label "array 1-$N" "$lj"
    echo "${GRN}labelling${RST} job $lj"
fi

if [[ " $STAGES " == *" build "* ]]; then
    dep=()
    [ -n "$lj" ] && dep=(--dependency=afterok:"$lj" --kill-on-invalid-dep=yes)
    bj=$(sbatch --parsable "${COMMON[@]}" --partition="$BUILD_PARTITION" --time="$BUILD_TIME" \
         "${dep[@]}" "$PIPE/data_pipeline/build_datasets.sbatch")
    log_job build "datasets" "$bj"
    echo "${GRN}build${RST}      job $bj"
fi

echo "${DIM}track progress with: bash track.sh   (or bash track.sh watch)${RST}"
