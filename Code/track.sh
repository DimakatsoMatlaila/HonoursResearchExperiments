#!/bin/bash
cd "$(dirname "$0")"
source ./config.sh
if [ "${1:-}" = "watch" ]; then
    secs="${2:-60}"
    while true; do
        clear
        "$PY" common/track.py
        echo; echo "${DIM}refreshing every ${secs}s, ctrl-c to stop${RST}"
        sleep "$secs"
    done
else
    "$PY" common/track.py "$@"
fi
