#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" == "token_here" ]; then
  echo "Error: GEMINI_API_KEY is set to the default value 'token_here' or is unset. Please update it in RunPod's environment variables or set it on your own."
  exit 1
else
  echo "GEMINI_API_KEY is set."
fi

echo "By running this script you're accepting Conda's TOS, if you do not accept those, please stop the script by clicking CTRL c"
sleep 5

REPO_DIR="/TripleX"
REPO_URL="https://github.com/Hearmeman24/TripleX.git"

if [ ! -d "$REPO_DIR" ]; then
    echo "Repository not found. Cloning..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository already exists. Skipping clone."
fi

# Define variables
CONDA_ENV_NAME="TripleX"
CONDA_ENV_PATH="/tmp/TripleX_miniconda/envs/$CONDA_ENV_NAME"
SCRIPT_PATH="/TripleX/captioners/gemini.py"
WORKING_DIR="$NETWORK_VOLUME/video_dataset_here"
REQUIREMENTS_PATH="/TripleX/requirements.txt"
CONDA_DIR="/tmp/TripleX_miniconda"

echo "Starting process..."

# Check if conda is already installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Conda not found. Installing Miniconda..."
    MINICONDA_PATH="/tmp/triplex/miniconda.sh"
    mkdir -p "/tmp/triplex"
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $MINICONDA_PATH
    bash $MINICONDA_PATH -b -p $CONDA_DIR
    rm $MINICONDA_PATH
    echo "Miniconda installed successfully."
else
    echo "Found existing Miniconda installation."
fi

# Initialize conda
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Check if environment exists
echo "Listing conda environments:"
conda env list

# Modify the check to be more explicit
if [ -d "$CONDA_DIR/envs/$CONDA_ENV_NAME" ]; then
    echo "Environment $CONDA_ENV_NAME exists in directory."
    conda activate $CONDA_ENV_NAME
else
    echo "Creating conda environment: $CONDA_ENV_NAME"
    conda create -y -n $CONDA_ENV_NAME python=3.10

    # Activate the environment
    source $CONDA_DIR/bin/activate $CONDA_ENV_NAME

    # Install dependencies from requirements.txt
    echo "Installing dependencies from requirements.txt..."
    if [ -f "$REQUIREMENTS_PATH" ]; then
        pip install -r $REQUIREMENTS_PATH
        pip install torchvision
    else
        echo "Warning: Requirements file not found at $REQUIREMENTS_PATH"
    fi
fi

# CUDA compatibility check
check_cuda_compatibility() {
    python << 'PYTHON_EOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        # Try a simple CUDA operation to test kernel compatibility
        x = torch.randn(1, device='cuda')
        y = x * 2
        print("CUDA compatibility check passed")
    else:
        print("\n" + "="*70)
        print("CUDA NOT AVAILABLE")
        print("="*70)
        print("\nCUDA is not available on this system.")
        print("This script requires CUDA to run.")
        print("\nSOLUTION:")
        print("  Please deploy with CUDA 12.8 when selecting your GPU on RunPod")
        print("  This template requires CUDA 12.8")
        print("\n" + "="*70)
        sys.exit(1)
except RuntimeError as e:
    error_msg = str(e).lower()
    if "no kernel image" in error_msg or "cuda error" in error_msg:
        print("\n" + "="*70)
        print("CUDA KERNEL COMPATIBILITY ERROR")
        print("="*70)
        print("\nThis error occurs when your GPU architecture is not supported")
        print("by the installed CUDA kernels. This typically happens when:")
        print("  • Your GPU model is older or different from what was expected")
        print("  • The PyTorch/CUDA build doesn't include kernels for your GPU")
        print("\nSOLUTIONS:")
        print("  1. Use a newer GPU model (recommended):")
        print("     • H100 or H200 GPUs are recommended for best compatibility")
        print("  2. Ensure correct CUDA version:")
        print("     • Filter for CUDA 12.8 when selecting your GPU on RunPod")
        print("     • This template requires CUDA 12.8")
        print("\n" + "="*70)
        sys.exit(1)
    else:
        raise
PYTHON_EOF
    if [ $? -ne 0 ]; then
        exit 1
    fi
}

check_cuda_compatibility

# Run the Python script
echo "Running gemini.py script..."
python $SCRIPT_PATH --dir "$WORKING_DIR" --max_frames 1
echo "video captioning complete"

echo "Script execution completed successfully."
echo "The conda environment '$CONDA_ENV_NAME' is preserved for future use."