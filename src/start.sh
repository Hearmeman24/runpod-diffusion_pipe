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

    # Move diffusion_pipe if it exists in root to working directory
    if [ -d "/diffusion_pipe" ]; then
        mv /diffusion_pipe "$NETWORK_VOLUME/"
    fi

    # Set up directory structure
    DIFF_PIPE_DIR="$NETWORK_VOLUME/diffusion_pipe"

    # Move training scripts and utilities
    if [ -d "$NETWORK_VOLUME/runpod-diffusion_pipe/start_training_scripts" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/start_training_scripts" "$NETWORK_VOLUME/"
    fi

    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/txt_replace.py" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/txt_replace.py" "$NETWORK_VOLUME/"
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

# Update dataset.toml path to use the working directory
if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml" ]; then
    sed -i "s|path = '/home/anon/data/images/grayscale'|path = '$NETWORK_VOLUME/image_dataset_here'|" "$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
fi

# Create models directory in the working directory
mkdir -p "$NETWORK_VOLUME/models"

# Download Wan 1.3B model if requested
if [ "$download_wan13" == "true" ]; then
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
    fi
    echo "Downloading Wan 1.3B model..."
    mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B"
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
    echo "Finished downloading Wan 1.3B model"
fi

# Download Wan 14B T2V model if requested
if [ "$download_wan14B_t2v" == "true" ]; then
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
    fi
    echo "Downloading Wan 14B T2V model..."
    mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B"
    huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
    echo "Finished downloading Wan 14B T2V model"
fi

# Download Wan 14B I2V model if requested
if [ "$download_wan14B_i2v_480p" == "true" ]; then
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
    fi
    echo "Downloading Wan 14B I2V model..."
    mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P"
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
    echo "Finished downloading Wan 14B I2V model"
fi

# Download base SDXL if requested
if [ "$download_base_sdxl" == "true" ]; then
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
    fi
    echo "Downloading Base SDXL..."
    huggingface-cli download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir "$NETWORK_VOLUME/models/" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
    echo "Finished downloading base SDXL"
fi

# Download Flux if requested
if [ "$download_flux" == "true" ]; then
    if [ -z "$HUGGING_FACE_TOKEN" ] || [ "$HUGGING_FACE_TOKEN" == "token_here" ]; then
        echo "Error: HUGGING_FACE_TOKEN is set to the default value 'token_here' or doesn't exist. Please update it in RunPod's environment variables."
        exit 1
    fi

    echo "HUGGING_FACE_TOKEN is set."
    if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml" ]; then
        mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
    fi
    echo "Downloading Flux..."
    mkdir -p "$NETWORK_VOLUME/models/flux"
    huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "$NETWORK_VOLUME/models/flux" --repo-type model --token "$HUGGING_FACE_TOKEN" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
    echo "Finished downloading Flux"
fi

# Clean up any redundant toml files in root
rm -f /*.toml

echo "Setup complete! All files are organized in $NETWORK_VOLUME"
echo "Jupyter Lab is running and accessible via the web interface"

sleep infinity