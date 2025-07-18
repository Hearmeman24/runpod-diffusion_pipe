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

# Run the Python script
echo "Running gemini.py script..."
python $SCRIPT_PATH --dir "$WORKING_DIR" --max_frames 1
echo "video captioning complete"

echo "Script execution completed successfully."
echo "The conda environment '$CONDA_ENV_NAME' is preserved for future use."