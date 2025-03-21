#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# This is in case there's any special installs or overrides that needs to occur when starting the machine before starting ComfyUI
if [ -f "/workspace/additional_params.sh" ]; then
    chmod +x /workspace/additional_params.sh
    echo "Executing additional_params.sh..."
    /workspace/additional_params.sh
else
    echo "additional_params.sh not found in /workspace. Skipping..."
fi

# Set the network volume path
NETWORK_VOLUME="/workspace"

# Check if NETWORK_VOLUME exists; if not, use root directory instead
if [ ! -d "$NETWORK_VOLUME" ]; then
    echo "NETWORK_VOLUME directory '$NETWORK_VOLUME' does not exist. You are NOT using a network volume. Setting NETWORK_VOLUME to '/' (root directory)."
    NETWORK_VOLUME="/"
    echo "NETWORK_VOLUME directory doesn't exist. Starting JupyterLab on root directory..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/ &
else
    echo "NETWORK_VOLUME directory exists. Starting JupyterLab..."
    jupyter-lab --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True --notebook-dir=/workspace &
fi

git clone https://github.com/Hearmeman24/TripleX.git
cd TripleX
pip install -r requirements.txt
apt-get update
apt-get install ffmpeg
cd $NETWORK_VOLUME


# Workspace as main working directory
echo "cd $NETWORK_VOLUME" >> ~/.bashrc
mkdir -p $NETWORK_VOLUME/dataset_here
sed -i "s|path = '/home/anon/data/images/grayscale'|path = '/dataset_here'|" $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml


if [ "$download_wan13" == "true" ]; then
  mv $NETWORK_VOLUME/wan13_video.toml $NETWORK_VOLUME/diffusion_pipe/examples
  echo "Downloading Wan 1.3B model"
  mkdir -p /Wan/Wan2.1-T2V-1.3B
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /Wan/Wan2.1-T2V-1.3B 2>&1 | tee download_log.txt
  echo "Finished downloading Wan 1.3B model"

fi

if [ "$download_wan14B" == "true" ]; then
  mv $NETWORK_VOLUME/wan14b_video.toml $NETWORK_VOLUME/diffusion_pipe/examples
  echo "Downloading Wan 14B model"
  mkdir -p /Wan/Wan2.1-T2V-14B
  huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir /Wan/Wan2.1-T2V-14B 2>&1 | tee download_log.txt
  echo "Finished downloading Wan 14B model"
fi

if [ "$download_wan14B_i2v_480p" == "true" ]; then
  mv $NETWORK_VOLUME/wan14b_video.toml $NETWORK_VOLUME/diffusion_pipe/examples
  echo "Downloading Wan 14B I2V model"
  mkdir -p /Wan/Wan2.1-I2V-14B-480P
  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /Wan/Wan2.1-I2V-14B-480P 2>&1 | tee download_log.txt
  echo "Finished downloading Wan 14B model"
fi

sleep infinity
