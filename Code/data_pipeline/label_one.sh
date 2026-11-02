#!/bin/bash
set -uo pipefail
source "$(dirname "$0")/../config.sh"
rel="$1"
profile="$(basename "$(dirname "$rel")")"
fname="$(basename "$rel")"
base="${fname%.*}"

case "$fname" in
    internal_*) side="internal" ;;
    external_*) side="external" ;;
    *) echo "[$(date +%T)] skipping $rel (not internal_*/external_*)"; exit 0 ;;
esac

c2=$(awk -v p="$profile" '!/^[[:space:]]*#/ && NF>=2 && $1==p {print $2; exit}' "$C2_IPS_FILE")
[ -n "$c2" ] || c2=$(awk '!/^[[:space:]]*#/ && NF>=2 && $1=="*" {print $2; exit}' "$C2_IPS_FILE")
[ -n "$c2" ] || { echo "no C2 IP for $profile in $C2_IPS_FILE"; mark_failed label "$rel"; exit 1; }
[ "$profile" = "normal" ] && c2="0.0.0.0"

final="$LABELED/$profile/${base}.csv"
tmp="$LABELED/$profile/.tmp_${base}.csv"
if [ -f "$final" ]; then
    echo "[$(date +%T)] already labeled: $rel"
    exit 0
fi

mkdir -p "$LABELED/$profile" 2>/dev/null || true
sleep 1
[ -d "$LABELED/$profile" ] || { echo "cannot create $LABELED/$profile"; mark_failed label "$rel"; exit 1; }

if [ -n "$REUSE_LABELED_FROM" ] && [ -f "$REUSE_LABELED_FROM/$profile/${base}.csv" ]; then
    echo "[$(date +%T)] reusing old labeled csv for $rel"
    cp "$REUSE_LABELED_FROM/$profile/${base}.csv" "$tmp" && mv "$tmp" "$final" && exit 0
    mark_failed label "$rel"; exit 1
fi

[ -f "$CLEAN/$rel" ] || { echo "cleaned file missing: $CLEAN/$rel"; mark_failed label "$rel"; exit 1; }

echo "[$(date +%T)] $(hostname): labeling $rel (profile=$profile side=$side c2=$c2)"
rm -f "$tmp"
if ! "$PY" "$PIPE/common/flow_features.py" "$CLEAN/$rel" "$tmp" "$profile" "$side" "$c2"; then
    echo "[$(date +%T)] FAILED: $rel"
    rm -f "$tmp"
    mark_failed label "$rel"
    exit 1
fi
mv "$tmp" "$final"
echo "[$(date +%T)] $(hostname): done $rel"
