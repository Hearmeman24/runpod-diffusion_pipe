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

    if [ "$IS_DEV" == "true" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/qwen_image_musubi_training" "$NETWORK_VOLUME/" 2>/dev/null || true
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/z_image_musubi_training" "$NETWORK_VOLUME/" 2>/dev/null || true
    fi

    if [ -d "/diffusion_pipe" ]; then
        mv /diffusion_pipe "$NETWORK_VOLUME/"
    fi

    DIFF_PIPE_DIR="$NETWORK_VOLUME/diffusion_pipe"

    if [ -d "$DIFF_PIPE_DIR" ] && [ -d "$DIFF_PIPE_DIR/.git" ]; then
        cd "$DIFF_PIPE_DIR" || exit 1
        git pull >> "$STARTUP_LOG" 2>&1 || true
        cd "$NETWORK_VOLUME" || exit 1
    fi

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
    fi
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

run_quiet "torch"          pip install torch torchvision torchaudio
run_quiet "transformers"   pip install transformers -U
run_quiet "huggingface"    pip install --upgrade "huggingface_hub[cli]"
run_quiet "peft"           pip install --upgrade "peft>=0.17.0"
run_quiet "deepspeed"      pip install --upgrade "deepspeed>=0.17.6"
run_quiet "diffusers"      bash -c "pip uninstall -y diffusers && pip install git+https://github.com/huggingface/diffusers"

if [ "$download_triton" == "true" ]; then
    run_quiet "triton" pip install triton
fi

# ============================================================
# [4/4] Starting JupyterLab
# ============================================================
status_msg "[4/4] Starting JupyterLab..."

jupyter-lab --ip=0.0.0.0 --allow-root --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True \
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
