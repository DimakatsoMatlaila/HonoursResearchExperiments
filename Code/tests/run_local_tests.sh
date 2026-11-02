#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
T="${1:-/tmp/c2test}"
rm -rf "$T"
mkdir -p "$T/raw"
export PY="${PY:-python3}" RAW="$T/raw" WORK="$T/work"

python3 "$HERE/tests/make_synthetic_data.py" "$RAW"
source "$HERE/config.sh"
bash "$HERE/data_pipeline/make_manifest.sh" > /dev/null

while read -r rel; do bash "$HERE/data_pipeline/clean_one.sh" "$rel" > /dev/null; done < "$MANIFEST"
while read -r rel; do bash "$HERE/data_pipeline/label_one.sh" "$rel" | tail -1; done < "$MANIFEST"
"$PY" "$HERE/common/build_datasets.py" "$LABELED" "$DATASETS"

for m in $MODELS; do
    for v in mixed https; do
        echo "---- $m $v"
        "$PY" "$HERE/models/$m/train.py" --data "$DATASETS/internal_dataset.csv" \
            --run-dir "$T/results/$m/internal_$v" --name internal --variant "$v" --smoke > "$T/$m.$v.log" 2>&1 \
            || { tail -30 "$T/$m.$v.log"; exit 1; }
        test -f "$T/results/$m/internal_$v/DONE" && echo "ok"
    done
done
echo "all local tests passed"
