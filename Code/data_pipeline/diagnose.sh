#!/bin/bash
source "$(dirname "$0")/../config.sh"
"$PY" "$PIPE/common/diagnose_c2_ip.py" "$LABELED" "$1"
