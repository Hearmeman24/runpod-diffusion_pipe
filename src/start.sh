#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Check if workspace exists and set network volume accordingly
if [ ! -d "/workspace" ]; then
    echo "NETWORK_VOLUME directory '/workspace' does not exist. You are NOT using a network volume. Setting NETWORK_VOLUME to '/diffusion_pipe_working_folder' (root directory)."
    mkdir -p "/diffusion_pipe_working_folder"
    NETWORK_VOLUME="/diffusion_pipe_working_folder"
else
    echo "Network volume detected at /workspace. Using /workspace/diffusion_pipe_working_folder as working directory."
    mkdir -p "/workspace/diffusion_pipe_working_folder"
    NETWORK_VOLUME="/workspace/diffusion_pipe_working_folder"
fi
export NETWORK_VOLUME

echo "cd $NETWORK_VOLUME" >> /root/.bashrc

#cd "$NETWORK_VOLUME/diffusion_pipe_working_folder/diffusion-pipe" || exit 1
#git pull || true
#cd "$NETWORK_VOLUME" || exit 1

# GPU detection for optimized flash-attn build
# Returns architecture in FLASH_ATTN_CUDA_ARCHS format (e.g., "90" for sm_90)
detect_cuda_arch() {
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
    
    # Save GPU name for other scripts to check
    echo "$gpu_name" > /tmp/detected_gpu
    
    case "$gpu_name" in
        # Blackwell Data Center (sm_100)
        *B100*|*B200*|*GB200*)
            echo "blackwell" > /tmp/gpu_arch_type
            echo "100"
            ;;
        # Blackwell Consumer/Pro (sm_120)
        *5090*|*5080*|*5070*|*5060*|*PRO*6000*Blackwell*)
            echo "blackwell" > /tmp/gpu_arch_type
            echo "120"
            ;;
        # Hopper (sm_90)
        *H100*|*H200*)
            echo "hopper" > /tmp/gpu_arch_type
            echo "90"
            ;;
        # Ada Lovelace (sm_89)
        *L4*|*L40*|*4090*|*4080*|*4070*|*4060*|*PRO*6000*Ada*)
            echo "ada" > /tmp/gpu_arch_type
            echo "89"
            ;;
        # Ampere (sm_86)
        *A10*|*A40*|*A6000*|*A5000*|*A4000*|*3090*|*3080*|*3070*|*3060*)
            echo "ampere" > /tmp/gpu_arch_type
            echo "86"
            ;;
        # Ampere Data Center (sm_80)
        *A100*)
            echo "ampere" > /tmp/gpu_arch_type
            echo "80"
            ;;
        # Turing (sm_75)
        *T4*|*2080*|*2070*|*2060*)
            echo "turing" > /tmp/gpu_arch_type
            echo "75"
            ;;
        # Volta (sm_70)
        *V100*)
            echo "volta" > /tmp/gpu_arch_type
            echo "70"
            ;;
        # Default: build for common modern architectures
        *)
            echo "unknown" > /tmp/gpu_arch_type
            echo "80;86;89;90"
            ;;
    esac
}

# Install flash-attn
# Strategy: Try prebuilt wheel first (fast) in foreground, fall back to building from source in background if needed
echo "Installing flash-attn..."
mkdir -p "$NETWORK_VOLUME/logs"

# Detect GPU and set optimal CUDA architecture
DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
CUDA_ARCH=$(detect_cuda_arch)
echo "Detected GPU: $DETECTED_GPU"
echo "Using CUDA architecture: $CUDA_ARCH"

# Specify the exact prebuilt wheel URL here
# Get wheels from: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
# Format: flash_attn-{version}+cu{cuda}torch{torch}-cp{py}-cp{py}-linux_x86_64.whl
FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"

# If no wheel URL specified, leave empty to build from source
# Example: FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.7.4+cu128torch2.7-cp312-cp312-linux_x86_64.whl"

WHEEL_INSTALLED=false

# Try prebuilt wheel if URL is provided (in foreground)
if [ -n "$FLASH_ATTN_WHEEL_URL" ]; then
    echo "Attempting to download prebuilt flash-attn wheel..."
    echo "  URL: $FLASH_ATTN_WHEEL_URL"
    
    cd /tmp
    WHEEL_NAME=$(basename "$FLASH_ATTN_WHEEL_URL")
    
    if wget -q -O "$WHEEL_NAME" "$FLASH_ATTN_WHEEL_URL" 2>&1; then
        echo "  Wheel downloaded successfully!"
        echo "  Installing wheel..."
        if pip install "$WHEEL_NAME" 2>&1; then
            rm -f "$WHEEL_NAME"
            echo "✅ Successfully installed flash-attn from prebuilt wheel!"
            WHEEL_INSTALLED=true
            # Create marker file to indicate wheel was successfully installed
            touch /tmp/flash_attn_wheel_success
        else
            echo "  Wheel installation failed, will build from source."
            rm -f "$WHEEL_NAME"
        fi
    else
        echo "  Failed to download wheel, will build from source."
    fi
else
    echo "No prebuilt wheel URL specified (FLASH_ATTN_WHEEL_URL is empty)."
    echo "Will build flash-attn from source."
fi

# Fall back to building from source in background if wheel not installed
if [ "$WHEEL_INSTALLED" = false ]; then
    echo ""
    echo "⚠️  Starting flash-attn build from source in background..."
    echo "   This may take 3-10 minutes depending on your system."
    echo ""
    
    # Dynamically calculate MAX_JOBS for fallback build
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
    
    # Build from source in background
    (
        set -e
        
        # Detect GPU for optimized build
        DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
        CUDA_ARCH=$(detect_cuda_arch)
        
        echo "Build configuration:"
        echo "  GPU: $DETECTED_GPU"
        echo "  CUDA Architecture: sm_$CUDA_ARCH"
        
        # Ensure ninja is installed for fast builds
        pip install ninja packaging -q
        if ! ninja --version > /dev/null 2>&1; then
            pip uninstall -y ninja && pip install ninja
        fi
        
        # Build from source
        cd /tmp
        rm -rf flash-attention
        
        echo "Cloning flash-attention repository..."
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention
        
        # Set build optimization environment variables
        export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH"
        export MAX_JOBS=$OPTIMAL_JOBS
        export NVCC_THREADS=4
        
        echo "Building with optimizations:"
        echo "  FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
        echo "  MAX_JOBS=$MAX_JOBS"
        echo "  NVCC_THREADS=$NVCC_THREADS"
        
        python setup.py install
        
        # Clean up
        cd /tmp
        rm -rf flash-attention
        
        echo "✅ Successfully built and installed flash-attn from source!"
        
    ) > "$NETWORK_VOLUME/logs/flash_attn_install.log" 2>&1 &
    FLASH_ATTN_PID=$!
    echo "$FLASH_ATTN_PID" > /tmp/flash_attn_pid
    echo "flash-attn build started in background (PID: $FLASH_ATTN_PID)"
    echo "To monitor progress: tail -f $NETWORK_VOLUME/logs/flash_attn_install.log"
fi

# Start Jupyter Lab with the working folder as the root directory
# This puts users directly in their working environment and hides system files
jupyter-lab --ip=0.0.0.0 --allow-root --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True \
    --notebook-dir="$NETWORK_VOLUME" &

# Move repository files to the working directory
if [ -d "/tmp/runpod-diffusion_pipe" ]; then
    # Move the entire repository to working directory
    mv /tmp/runpod-diffusion_pipe "$NETWORK_VOLUME/"
    mv "$NETWORK_VOLUME/runpod-diffusion_pipe/Captioning" "$NETWORK_VOLUME/"
    mv "$NETWORK_VOLUME/runpod-diffusion_pipe/wan2.2_lora_training" "$NETWORK_VOLUME/"
    
    # Only move Qwen folder if IS_DEV is set to true
    if [ "$IS_DEV" == "true" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/qwen_image_musubi_training" "$NETWORK_VOLUME/" 2>/dev/null || true
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/z_image_musubi_training" "$NETWORK_VOLUME/" 2>/dev/null || true
    fi


    # Move diffusion_pipe if it exists in root to working directory
    if [ -d "/diffusion_pipe" ]; then
        mv /diffusion_pipe "$NETWORK_VOLUME/"
    fi

    # Set up directory structure
    DIFF_PIPE_DIR="$NETWORK_VOLUME/diffusion_pipe"
    
    # Pull latest changes from diffusion_pipe repository
    if [ -d "$DIFF_PIPE_DIR" ] && [ -d "$DIFF_PIPE_DIR/.git" ]; then
        echo "Pulling latest changes from diffusion_pipe repository..."
        cd "$DIFF_PIPE_DIR" || exit 1
        git pull || echo "Warning: Failed to pull latest changes from diffusion_pipe repository"
        cd "$NETWORK_VOLUME" || exit 1
    else
        echo "Warning: diffusion_pipe directory not found or not a git repository. Skipping git pull."
    fi


    echo "Updating TOML file paths..."
    TOML_DIR="$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files"
    if [ -d "$TOML_DIR" ]; then
        # Update paths in TOML files to use NETWORK_VOLUME
        for toml_file in "$TOML_DIR"/*.toml; do
            if [ -f "$toml_file" ]; then
                echo "Processing: $(basename "$toml_file")"
                # Create backup
                cp "$toml_file" "$toml_file.backup"

                # Update various path patterns - replace absolute paths with NETWORK_VOLUME paths
                sed -i "s|diffusers_path = '/models/|diffusers_path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|ckpt_path = '/Wan/|ckpt_path = '$NETWORK_VOLUME/models/Wan/|g" "$toml_file"
                sed -i "s|checkpoint_path = '/models/|checkpoint_path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|output_dir = '/data/|output_dir = '$NETWORK_VOLUME/training_outputs/|g" "$toml_file"
                sed -i "s|output_dir = '/training_outputs/|output_dir = '$NETWORK_VOLUME/training_outputs/|g" "$toml_file"

                # Handle commented paths too
                sed -i "s|#transformer_path = '/models/|#transformer_path = '$NETWORK_VOLUME/models/|g" "$toml_file"

                # Z-Image model paths
                sed -i "s|diffusion_model = '/models/|diffusion_model = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|vae = '/models/|vae = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|{path = '/models/|{path = '$NETWORK_VOLUME/models/|g" "$toml_file"
                sed -i "s|merge_adapters = \['/models/|merge_adapters = ['$NETWORK_VOLUME/models/|g" "$toml_file"

                echo "Updated paths in: $(basename "$toml_file")"
            fi
        done
    fi

    # Move training scripts and utilities
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/interactive_start_training.sh" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/interactive_start_training.sh" "$NETWORK_VOLUME/"
        chmod +x "$NETWORK_VOLUME/interactive_start_training.sh"
    fi

    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/HowToUse.txt" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/HowToUse.txt" "$NETWORK_VOLUME/"
    fi

    # Set up send_lora.sh script
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/send_lora.sh" ]; then
        chmod +x "$NETWORK_VOLUME/runpod-diffusion_pipe/send_lora.sh"
        cp "$NETWORK_VOLUME/runpod-diffusion_pipe/send_lora.sh" /usr/local/bin/
    fi

    # Clean up examples and move dataset.toml
    if [ -d "$NETWORK_VOLUME/diffusion_pipe/examples" ]; then
        rm -rf "$NETWORK_VOLUME/diffusion_pipe/examples"/*
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
        fi
    fi
fi

# Install Triton if requested
if [ "$download_triton" == "true" ]; then
    echo "Installing Triton..."
    pip install triton
fi

# Create dataset directories in the working directory
mkdir -p "$NETWORK_VOLUME/image_dataset_here"
mkdir -p "$NETWORK_VOLUME/video_dataset_here"
mkdir -p "$NETWORK_VOLUME/logs"
# Update dataset.toml path to use the working directory
if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml" ]; then
    sed -i "s|path = '/home/anon/data/images/grayscale'|path = '$NETWORK_VOLUME/image_dataset_here'|" "$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
fi

echo "Installing torch"
pip install torch torchvision torchaudio

echo "Upgrading transformers package..."
pip install transformers -U

echo "Installing huggingface-cli..."
pip install --upgrade "huggingface_hub[cli]"

echo "Upgrading peft package..."
pip install --upgrade "peft>=0.17.0"

echo "Updating diffusers package..."
pip uninstall -y diffusers
pip install git+https://github.com/huggingface/diffusers

echo "================================================"
echo "✅ Jupyter Lab is running and accessible via the web interface"
echo "================================================"

sleep infinity