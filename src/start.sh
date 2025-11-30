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

                # Handle commented paths too
                sed -i "s|#transformer_path = '/models/|#transformer_path = '$NETWORK_VOLUME/models/|g" "$toml_file"

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