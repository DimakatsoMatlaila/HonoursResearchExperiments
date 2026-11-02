#!/bin/bash
set -uo pipefail
source "$(dirname "$0")/../config.sh"
rel="$1"
dest="$CLEAN/$rel"
destdir="$(dirname "$dest")"
tmp="$destdir/.tmp_$(basename "$dest")"

if [ -f "$dest" ]; then
    echo "[$(date +%T)] already cleaned: $rel"
    exit 0
fi

mkdir -p "$destdir" 2>/dev/null || true
sleep 1
[ -d "$destdir" ] || { echo "cannot create $destdir"; mark_failed clean "$rel"; exit 1; }

echo "[$(date +%T)] $(hostname): cleaning $rel"
rm -f "$tmp"
if ! "$PY" "$PIPE/common/filter_pcaps.py" "$RAW/$rel" "$tmp"; then
    echo "[$(date +%T)] FAILED: $rel"
    rm -f "$tmp"
    mark_failed clean "$rel"
    exit 1
fi
mv "$tmp" "$dest"
echo "[$(date +%T)] $(hostname): done $rel"
