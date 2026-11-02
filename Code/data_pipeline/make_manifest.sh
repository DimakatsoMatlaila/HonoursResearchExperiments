#!/bin/bash
source "$(dirname "$0")/../config.sh"
mkdir -p "$PIPE/manifests"
(cd "$RAW" && find . -type f \( -name '*.pcap' -o -name '*.pcapng' \) | sed 's|^\./||' | sort) > "$MANIFEST"
echo "${BLU}manifest:${RST} $(wc -l < "$MANIFEST") captures -> $MANIFEST"
bad=$(grep -vE '(^|/)(internal|external)_[^/]*$' "$MANIFEST" || true)
if [ -n "$bad" ]; then
    echo "${YEL}these are not named internal_*/external_* and will be skipped when labelling:${RST}"
    echo "$bad"
fi
