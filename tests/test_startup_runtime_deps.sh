#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_SCRIPT="$PROJECT_DIR/src/start.sh"

fail() {
    echo "FAIL: $1"
    exit 1
}

grep -q 'pip install.*protobuf<7' "$START_SCRIPT" \
    || fail "start.sh must install protobuf<7 at runtime"

grep -q 'pip install.*comfy_aimdo' "$START_SCRIPT" \
    || fail "start.sh must install comfy_aimdo at runtime"

deps_line=$(grep -n 'protobuf<7' "$START_SCRIPT" | head -n1 | cut -d: -f1)
jupyter_line=$(grep -n 'jupyter-lab' "$START_SCRIPT" | head -n1 | cut -d: -f1)

[ -n "$deps_line" ] || fail "could not locate runtime dependency install line"
[ -n "$jupyter_line" ] || fail "could not locate jupyter startup line"
[ "$deps_line" -lt "$jupyter_line" ] \
    || fail "runtime dependencies must install before Jupyter starts"

echo "PASS: startup runtime dependency installs are present"
