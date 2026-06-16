#!/usr/bin/env bash

# ============================================================
# Helper functions for clean user-facing output
# ============================================================

STARTUP_LOG=""

status_msg() {
    echo ""
    echo "  $1"
}

# Run a command quietly, logging output to STARTUP_LOG.
# Shows "Still working..." every 10 seconds.
# On failure, prints a warning with the log path.
run_quiet() {
    local label="$1"
    shift

    # Start a background heartbeat that prints every 10 seconds
    (
        while true; do
            sleep 10
            echo "       Still working..."
        done
    ) &
    local heartbeat_pid=$!

    # Run the actual command, suppress output to log
    "$@" >> "$STARTUP_LOG" 2>&1
    local exit_code=$?

    # Kill the heartbeat
    kill "$heartbeat_pid" 2>/dev/null
    wait "$heartbeat_pid" 2>/dev/null

    if [ $exit_code -ne 0 ]; then
        echo "       Warning: $label may have failed. Check $STARTUP_LOG for details."
    fi

    return $exit_code
}

# ============================================================
# Use libtcmalloc for better memory management
# ============================================================
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ============================================================
# Detect workspace and set NETWORK_VOLUME
# ============================================================
if [ ! -d "/workspace" ]; then
    mkdir -p "/diffusion_pipe_working_folder"
    NETWORK_VOLUME="/diffusion_pipe_working_folder"
else
    mkdir -p "/workspace/diffusion_pipe_working_folder"
    NETWORK_VOLUME="/workspace/diffusion_pipe_working_folder"
fi
export NETWORK_VOLUME

echo "cd $NETWORK_VOLUME" >> /root/.bashrc

mkdir -p "$NETWORK_VOLUME/logs"
STARTUP_LOG="$NETWORK_VOLUME/logs/startup.log"
echo "--- Startup log $(date) ---" > "$STARTUP_LOG"

# diffusion-pipe lives on the persistent volume and is (re)asserted on every boot, so its
# path must be known unconditionally — NOT only inside the first-boot staging guard below.
DIFF_PIPE_DIR="$NETWORK_VOLUME/diffusion_pipe"
export DIFF_PIPE_DIR

# Degradation sentinel: cleared exactly ONCE here, at boot start. Every stage below only
# APPENDS reasons to it; nothing else clears it. interactive_start_training.sh refuses to
# launch training if it exists at training time (the real gate — start.sh ends in sleep
# infinity and never launches training itself).
rm -f /tmp/ENV_DEGRADED

# ============================================================
# GPU detection (quiet - only writes to files and returns value)
# ============================================================
detect_cuda_arch() {
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
    echo "$gpu_name" > /tmp/detected_gpu

    case "$gpu_name" in
        *B100*|*B200*|*GB200*)
            echo "blackwell" > /tmp/gpu_arch_type; echo "100" ;;
        *5090*|*5080*|*5070*|*5060*|*PRO*6000*Blackwell*)
            echo "blackwell" > /tmp/gpu_arch_type; echo "120" ;;
        *H100*|*H200*)
            echo "hopper" > /tmp/gpu_arch_type; echo "90" ;;
        *L4*|*L40*|*4090*|*4080*|*4070*|*4060*|*PRO*6000*Ada*)
            echo "ada" > /tmp/gpu_arch_type; echo "89" ;;
        *A10*|*A40*|*A6000*|*A5000*|*A4000*|*3090*|*3080*|*3070*|*3060*)
            echo "ampere" > /tmp/gpu_arch_type; echo "86" ;;
        *A100*)
            echo "ampere" > /tmp/gpu_arch_type; echo "80" ;;
        *T4*|*2080*|*2070*|*2060*)
            echo "turing" > /tmp/gpu_arch_type; echo "75" ;;
        *V100*)
            echo "volta" > /tmp/gpu_arch_type; echo "70" ;;
        *)
            echo "unknown" > /tmp/gpu_arch_type; echo "80;86;89;90" ;;
    esac
}

DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
CUDA_ARCH=$(detect_cuda_arch)

# ============================================================
# Startup banner
# ============================================================
echo ""
echo "================================================"
echo "  Starting up..."
echo "  GPU: $DETECTED_GPU"
echo "================================================"

# ============================================================
# [1/4] Flash attention
# ============================================================
status_msg "[1/4] Installing flash attention..."

FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
WHEEL_INSTALLED=false

if [ -n "$FLASH_ATTN_WHEEL_URL" ]; then
    cd /tmp
    WHEEL_NAME=$(basename "$FLASH_ATTN_WHEEL_URL")

    if wget -q -O "$WHEEL_NAME" "$FLASH_ATTN_WHEEL_URL" >> "$STARTUP_LOG" 2>&1; then
        if pip install "$WHEEL_NAME" >> "$STARTUP_LOG" 2>&1; then
            rm -f "$WHEEL_NAME"
            WHEEL_INSTALLED=true
            touch /tmp/flash_attn_wheel_success
        else
            rm -f "$WHEEL_NAME"
        fi
    fi
fi

# Fall back to building from source in background if wheel not installed
if [ "$WHEEL_INSTALLED" = false ]; then
    echo "       Building from source in background (this may take a few minutes)..."

    CPU_CORES=$(nproc)
    CPU_JOBS=$(( CPU_CORES - 2 ))
    [ "$CPU_JOBS" -lt 4 ] && CPU_JOBS=4
    AVAILABLE_RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
    RAM_JOBS=$(( AVAILABLE_RAM_GB / 3 ))
    [ "$RAM_JOBS" -lt 4 ] && RAM_JOBS=4
    if [ "$CPU_JOBS" -lt "$RAM_JOBS" ]; then
        OPTIMAL_JOBS=$CPU_JOBS
    else
        OPTIMAL_JOBS=$RAM_JOBS
    fi

    (
        set -e
        DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
        CUDA_ARCH=$(detect_cuda_arch)

        pip install ninja packaging -q
        if ! ninja --version > /dev/null 2>&1; then
            pip uninstall -y ninja && pip install ninja
        fi

        cd /tmp
        rm -rf flash-attention
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention

        export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH"
        export MAX_JOBS=$OPTIMAL_JOBS
        export NVCC_THREADS=4

        python setup.py install

        cd /tmp
        rm -rf flash-attention
    ) > "$NETWORK_VOLUME/logs/flash_attn_install.log" 2>&1 &
    FLASH_ATTN_PID=$!
    echo "$FLASH_ATTN_PID" > /tmp/flash_attn_pid
fi

# ============================================================
# [2/4] Setting up workspace
# ============================================================
status_msg "[2/4] Setting up workspace..."

if [ -d "/tmp/runpod-diffusion_pipe" ]; then
    mv /tmp/runpod-diffusion_pipe "$NETWORK_VOLUME/"
    mv "$NETWORK_VOLUME/runpod-diffusion_pipe/Captioning" "$NETWORK_VOLUME/" 2>/dev/null || true
    mv "$NETWORK_VOLUME/runpod-diffusion_pipe/wan2.2_lora_training" "$NETWORK_VOLUME/" 2>/dev/null || true

    if [ -d "/diffusion_pipe" ]; then
        mv /diffusion_pipe "$NETWORK_VOLUME/"
    fi

    # NOTE: the diffusion-pipe pin / ComfyUI de-drift / gloo sed used to live HERE. They were
    # moved OUT to the always-run block below (after this guard closes) so they re-assert on
    # every boot, not just the boot that happens to stage /tmp/runpod-diffusion_pipe. DIFF_PIPE_DIR
    # is now defined unconditionally near the top of this script.

    TOML_DIR="$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files"
    if [ -d "$TOML_DIR" ]; then
        for toml_file in "$TOML_DIR"/*.toml; do
            if [ -f "$toml_file" ]; then
                cp "$toml_file" "$toml_file.backup"
                sed -i "s|diffusers_path = '/models/|diffusers_path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|ckpt_path = '/Wan/|ckpt_path = '$NETWORK_VOLUME/models/Wan/|g" "$toml_file"
                sed -i "s|checkpoint_path = '/models/|checkpoint_path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|output_dir = '/data/|output_dir = '$NETWORK_VOLUME/training_outputs/|g" "$toml_file"
                sed -i "s|output_dir = '/training_outputs/|output_dir = '$NETWORK_VOLUME/training_outputs/|g" "$toml_file"
                sed -i "s|#transformer_path = '/models/|#transformer_path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|diffusion_model = '/models/|diffusion_model = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|vae = '/models/|vae = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|{path = '/models/|{path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|merge_adapters = \['/models/|merge_adapters = ['$NETWORK_VOLUME/models/|g" "$toml_file"
            fi
        done
    fi

    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/interactive_start_training.sh" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/interactive_start_training.sh" "$NETWORK_VOLUME/"
        chmod +x "$NETWORK_VOLUME/interactive_start_training.sh"
    fi

    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/HowToUse.txt" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/HowToUse.txt" "$NETWORK_VOLUME/"
    fi

    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/send_lora.sh" ]; then
        chmod +x "$NETWORK_VOLUME/runpod-diffusion_pipe/send_lora.sh"
        cp "$NETWORK_VOLUME/runpod-diffusion_pipe/send_lora.sh" /usr/local/bin/
    fi

    if [ -d "$NETWORK_VOLUME/diffusion_pipe/examples" ]; then
        rm -rf "$NETWORK_VOLUME/diffusion_pipe/examples"/*
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
        fi
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset_qwen.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset_qwen.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
        fi
    fi
fi

# ============================================================
# ALWAYS-RUN: pin diffusion-pipe + de-drift its ComfyUI submodule + force the gloo backend.
# Runs on EVERY boot (keyed only on the repo existing on the volume), NOT just the boot that
# stages /tmp. This is what de-drifts a persistent volume that a prior boot pushed to upstream
# master. Each pin is a fixed sha, so re-running is idempotent.
# ============================================================
# Full 40-char object names — GitHub's smart-HTTP only serves a `want` on a FULL sha (or a ref),
# NOT an abbreviated one, so the targeted `git fetch origin <sha>` below must use the full form.
DIFF_PIPE_PIN="5aa65772168809346629d65a094d3a5523331669"
COMFYUI_PIN="f49bdb655707b97952dcef40e12e5af1f08d2007"

if [ -d "$DIFF_PIPE_DIR/.git" ]; then
    cd "$DIFF_PIPE_DIR" || exit 1

    # A bare `git fetch origin` does NOT deepen a shallow volume clone, so a later
    # `git reset --hard <sha>` would hard-error if the pin isn't in history. Fetch the EXACT
    # pinned object (works on shallow clones); unshallow, then a plain fetch, as fallbacks.
    git fetch --depth=1 origin "$DIFF_PIPE_PIN" >> "$STARTUP_LOG" 2>&1 \
        || git fetch --unshallow >> "$STARTUP_LOG" 2>&1 \
        || git fetch --quiet origin >> "$STARTUP_LOG" 2>&1 || true

    # A prior boot's gloo sed dirtied train.py (a TRACKED file); a dirty tree makes
    # `git reset --hard` refuse to move some paths. Discard working-tree edits FIRST, then
    # re-apply the sed below.
    git checkout -- . >> "$STARTUP_LOG" 2>&1 || true

    # Pin to the TESTED commit, or FAIL LOUDLY. Do NOT fall back to origin/main — that re-points
    # everything at drifting upstream and re-breaks the exact bug class we are fixing. If the pin
    # isn't fetchable, write the sentinel and stop touching the repo.
    if ! git reset --hard "$DIFF_PIPE_PIN" >> "$STARTUP_LOG" 2>&1; then
        echo "diffusion-pipe pin $DIFF_PIPE_PIN not fetchable — refusing to take latest" >> /tmp/ENV_DEGRADED
    else
        # The persistent volume's ComfyUI submodule may already be advanced past the gitlink
        # (a prior boot's now-deleted master loop). Plain `submodule update` will NOT move an
        # advanced/branch checkout back; --force does, and `sync` rewrites the URL from the
        # pinned .gitmodules first.
        git submodule sync --recursive >> "$STARTUP_LOG" 2>&1 || true
        git submodule update --init --recursive --force >> "$STARTUP_LOG" 2>&1 || true
        # Belt: fetch the EXACT gitlink object and check it out, pinning ComfyUI to v0.24.0
        # INDEPENDENTLY of which diffusion-pipe commit the superproject landed on (so nothing
        # can silently re-drift ComfyUI to a newer, scipy-breaking commit).
        git -C submodules/ComfyUI fetch --quiet origin "$COMFYUI_PIN" >> "$STARTUP_LOG" 2>&1 || true
        git -C submodules/ComfyUI checkout --quiet "$COMFYUI_PIN" >> "$STARTUP_LOG" 2>&1 \
            || echo "ComfyUI submodule not at gitlink $COMFYUI_PIN (de-drift failed)" >> /tmp/ENV_DEGRADED
    fi

    # Force the gloo backend for distributed init.
    # RunPod's H100 hosts now ship CUDA 13 host drivers; NCCL collectives (even the
    # single-rank metadata broadcast in dataset.py) SIGSEGV with the current
    # torch 2.9 / nccl 2.27 stack on those drivers, killing every training run.
    # Single-GPU LoRA training (--num_gpus=1) only needs trivial CPU broadcasts, which
    # gloo handles; gradient math is identical. (Diagnosed via the template sanity sweep, 2026-06-15.)
    # Grep-guarded so it is idempotent and survives the `git checkout -- .` above (which reverts it).
    if [ -f "$DIFF_PIPE_DIR/train.py" ] && ! grep -q 'dist_backend="gloo"' "$DIFF_PIPE_DIR/train.py"; then
        sed -i 's/deepspeed\.init_distributed()/deepspeed.init_distributed(dist_backend="gloo")/' "$DIFF_PIPE_DIR/train.py"
    fi

    cd "$NETWORK_VOLUME" || exit 1
fi

mkdir -p "$NETWORK_VOLUME/image_dataset_here"
mkdir -p "$NETWORK_VOLUME/video_dataset_here"

if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml" ]; then
    sed -i "s|path = '/home/anon/data/images/grayscale'|path = '$NETWORK_VOLUME/image_dataset_here'|" "$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
fi

# ============================================================
# [3/4] Fetching latest package updates
# ============================================================
status_msg "[3/4] Fetching latest updates..."

# Pinned overlay. The old block force-UPGRADED torch/transformers/hf/peft/deepspeed to latest and
# installed diffusers from git main on every boot — that daily drift is exactly what broke the
# template (diffusers main dropped txt_seq_lens -> Qwen TypeError; ComfyUI master pulled scipy in).
# We now install a single internally-consistent pinned set and honor diffusion-pipe's own
# requirements.txt as the source of truth for the long tail (deepspeed==0.18.4, scipy, etc.).
export PIP_INDEX_URL="https://download.pytorch.org/whl/cu128"
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"

# Constraints file applied to EVERY pip step so requirements.txt / the interactive script can't
# re-float these. torch trio: the cu128 build the flash-attn wheel ([1/4]) was compiled against.
# transformers: MUST stay on 4.x — transformers 5.x removed CLIPTextModel.text_model, which breaks
# diffusion-pipe's SDXL/CLIP loading (verified live on Phase-0 2026-06-16: 5.12.1 fails, 4.57.6 trains).
# peft 0.19.1 is the version that trains green with this set.
printf 'torch==2.9.1\ntorchvision==0.24.1\ntorchaudio==2.9.1\ntransformers==4.57.6\npeft==0.19.1\n' > /tmp/pins.txt
run_quiet "torch trio" pip install -c /tmp/pins.txt torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

# Honor diffusion-pipe's requirements.txt (single source of truth for the transitive set), but
# strip comfy-kitchen so its py3-none-any STUB wheel can never resolve here — we install the
# NATIVE comfy-kitchen explicitly below (fp8 requires the compiled build).
if [ -f "$DIFF_PIPE_DIR/requirements.txt" ]; then
    grep -v -i 'comfy-kitchen\|comfy_kitchen' "$DIFF_PIPE_DIR/requirements.txt" > /tmp/req_no_stub.txt
    run_quiet "requirements" pip install -c /tmp/pins.txt -r /tmp/req_no_stub.txt
else
    echo "requirements.txt missing at $DIFF_PIPE_DIR — diffusion-pipe deps not asserted" >> /tmp/ENV_DEGRADED
fi

# Native fp8 comfy-kitchen. --only-binary=:all: forbids an sdist build; combined with the
# grep-strip above (so the stub wheel is never a transitive option) this lands the compiled wheel.
run_quiet "comfy-kitchen (native)" pip install -c /tmp/pins.txt --only-binary=:all: comfy-kitchen==0.2.10

# Over-pin the packages requirements.txt leaves loose: diffusers (>=0.35.1 would re-pull main and
# re-break Qwen), transformers (unpinned -> 5.x breaks SDXL CLIP), peft, comfy-aimdo, and protobuf.
run_quiet "pinned overrides" pip install -c /tmp/pins.txt diffusers==0.38.0 transformers==4.57.6 peft==0.19.1 comfy_aimdo==0.4.10 "protobuf<7"

if [ "$download_triton" == "true" ]; then
    run_quiet "triton" pip install -c /tmp/pins.txt triton
fi

# ------------------------------------------------------------
# Boot diagnostics — APPEND-only to /tmp/ENV_DEGRADED (cleared once at boot start).
# These DIAGNOSE; the real gate is interactive_start_training.sh refusing to train if the
# sentinel exists (start.sh ends in sleep infinity and never launches training itself).
# ------------------------------------------------------------
# (a) torch is exactly the pinned cu128 build.
python -c "import torch,sys; v=torch.__version__; sys.exit(0 if v.startswith('2.9.1') and 'cu128' in v else 1)" 2>/dev/null \
    || echo "torch != 2.9.1+cu128 (got $(python -c 'import torch;print(torch.__version__)' 2>/dev/null))" >> /tmp/ENV_DEGRADED
# (b) deepspeed is the pinned version (caught a failed install / wrong resolution).
python -c "import deepspeed,sys; sys.exit(0 if deepspeed.__version__=='0.18.4' else 1)" 2>/dev/null \
    || echo "deepspeed != 0.18.4 (got $(python -c 'import deepspeed;print(deepspeed.__version__)' 2>/dev/null))" >> /tmp/ENV_DEGRADED
# (c) comfy-kitchen is the NATIVE fp8 build, not the py3-none-any stub. The native cp312-abi3 wheel
# ships a compiled .so, but in a SUBPACKAGE (backends/cuda/_C.abi3.so) — NOT the top dir. Probe
# RECURSIVELY with os.walk; a top-dir-only check false-flags the healthy native wheel as a stub
# (verified live on the Phase-0 pod 2026-06-16: native wheel installs _C.abi3.so under backends/cuda/).
python -c "import comfy_kitchen,os,sys; d=os.path.dirname(comfy_kitchen.__file__); sys.exit(0 if any(fn.endswith('.so') for _,_,fs in os.walk(d) for fn in fs) else 1)" 2>/dev/null \
    || echo "comfy_kitchen has no compiled .so anywhere in the package (stub wheel) — fp8 will silently degrade" >> /tmp/ENV_DEGRADED

if [ -f /tmp/ENV_DEGRADED ]; then
    { echo "ENV DEGRADED:"; cat /tmp/ENV_DEGRADED; } | tee -a "$STARTUP_LOG"
fi

# ============================================================
# [4/4] Starting JupyterLab
# ============================================================
status_msg "[4/4] Starting JupyterLab..."

jupyter-lab --ip=0.0.0.0 --allow-root --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True \
    --ServerApp.terminado_settings='{"shell_command": ["/bin/bash"]}' \
    --notebook-dir="$NETWORK_VOLUME" >> "$STARTUP_LOG" 2>&1 &

# ============================================================
# Ready!
# ============================================================
echo ""
echo "================================================"
echo ""
echo "  Template ready!"
echo "  Open JupyterLab from the RunPod web interface."
echo ""
echo "================================================"
echo ""

sleep infinity
