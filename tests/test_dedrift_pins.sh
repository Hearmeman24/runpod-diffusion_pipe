#!/bin/bash
set -u

# Regression test for D-shortsha: the always-run de-drift block fetches the pinned diffusion-pipe
# and ComfyUI commits by EXACT object name. GitHub's smart-HTTP only serves a `want` on a FULL
# 40-char sha (or a ref) — an abbreviated sha makes `git fetch origin <sha>` fail with
# "couldn't find remote ref", which silently breaks the targeted shallow-clone fetch and the
# ComfyUI gitlink-recovery belt. So both fetch-target pins MUST be full 40-char hex shas.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
START_SCRIPT="$PROJECT_DIR/src/start.sh"

fail() {
    echo "FAIL: $1"
    exit 1
}

# Extract the pin values assigned in start.sh.
diff_pipe_pin=$(grep -E '^DIFF_PIPE_PIN=' "$START_SCRIPT" | head -n1 | sed -E 's/^DIFF_PIPE_PIN="?([^"]*)"?.*/\1/')
comfyui_pin=$(grep -E '^COMFYUI_PIN=' "$START_SCRIPT" | head -n1 | sed -E 's/^COMFYUI_PIN="?([^"]*)"?.*/\1/')

[ -n "$diff_pipe_pin" ] || fail "DIFF_PIPE_PIN not found in start.sh"
[ -n "$comfyui_pin" ]   || fail "COMFYUI_PIN not found in start.sh"

# Both pins are used as `git fetch origin <pin>` targets, so each must be a full 40-char hex sha.
echo "$diff_pipe_pin" | grep -Eq '^[0-9a-f]{40}$' \
    || fail "DIFF_PIPE_PIN must be a full 40-char sha (GitHub rejects fetch of an abbreviated sha), got: $diff_pipe_pin"
echo "$comfyui_pin" | grep -Eq '^[0-9a-f]{40}$' \
    || fail "COMFYUI_PIN must be a full 40-char sha (GitHub rejects fetch of an abbreviated sha), got: $comfyui_pin"

echo "PASS: de-drift pins are full 40-char shas ($diff_pipe_pin, $comfyui_pin)"

# Optional live check (opt-in: RUN_NETWORK_TESTS=1) — proves the full shas are real, fetchable
# objects and that the de-drift block lands them against the real GitHub remotes from a shallow
# clone with a drifted ComfyUI submodule. Skipped by default so the suite stays offline/fast.
if [ "${RUN_NETWORK_TESTS:-0}" != "1" ]; then
    echo "SKIP (set RUN_NETWORK_TESTS=1 to run): live de-drift fetch against github.com"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone --quiet --depth=1 --recurse-submodules https://github.com/tdrussell/diffusion-pipe "$tmp/dp" \
    || fail "could not clone diffusion-pipe for live check"
cd "$tmp/dp" || fail "cd failed"

# Drift + gc the submodule so the gitlink object is not already present (the case the belt rescues).
git -C submodules/ComfyUI fetch --quiet --depth=1 origin master 2>/dev/null
git -C submodules/ComfyUI checkout --quiet FETCH_HEAD 2>/dev/null

git fetch --depth=1 origin "$diff_pipe_pin" >/dev/null 2>&1 \
    || git fetch --unshallow >/dev/null 2>&1 \
    || git fetch --quiet origin >/dev/null 2>&1 || true
git checkout -- . >/dev/null 2>&1 || true
git reset --hard "$diff_pipe_pin" >/dev/null 2>&1 \
    || fail "live: diffusion-pipe pin $diff_pipe_pin not fetchable/reset"
git submodule sync --recursive >/dev/null 2>&1 || true
git submodule update --init --recursive --force >/dev/null 2>&1 || true
git -C submodules/ComfyUI fetch --quiet origin "$comfyui_pin" >/dev/null 2>&1 || true
git -C submodules/ComfyUI checkout --quiet "$comfyui_pin" >/dev/null 2>&1 \
    || fail "live: ComfyUI submodule could not reach gitlink $comfyui_pin"

[ "$(git rev-parse HEAD)" = "$diff_pipe_pin" ] \
    || fail "live: diffusion-pipe HEAD is not the pin"
[ "$(git -C submodules/ComfyUI rev-parse HEAD)" = "$comfyui_pin" ] \
    || fail "live: ComfyUI submodule is not at the gitlink"

echo "PASS: live de-drift lands both pins against github.com (shallow clone + drifted submodule)"
