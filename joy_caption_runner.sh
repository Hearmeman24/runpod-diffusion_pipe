#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

REPO_DIR="/joy-caption-batch"
REPO_URL="https://github.com/MNeMoNiCuZ/joy-caption-batch.git"

if [ ! -d "$REPO_DIR" ]; then
    echo "Repository not found. Cloning..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "Repository already exists. Skipping clone."
fi

# Define variables
CONDA_ENV_NAME="joy_caption"
SCRIPT_PATH="/joy-caption-batch/batch-alpha2.py"
INPUT_DIR="/joy-caption-batch/input"
OUTPUT_DIR="/dataset_here"
REQUIREMENTS_PATH="/joy-caption-batch/requirements.txt"
CONDA_DIR="/tmp/miniconda"

echo "Starting process..."

# Make sure output directory exists
mkdir -p $OUTPUT_DIR

# Check if conda is already installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Conda not found. Installing Miniconda..."
    MINICONDA_PATH="/tmp/miniconda.sh"
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

# Check if environment exists
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating conda environment: $CONDA_ENV_NAME"
    conda create -y -n $CONDA_ENV_NAME python=3.10

    # Activate the environment
    conda activate $CONDA_ENV_NAME

    # Install dependencies from requirements.txt
    echo "Installing dependencies from requirements.txt..."
    if [ -f "$REQUIREMENTS_PATH" ]; then
        pip install -r $REQUIREMENTS_PATH
        pip install torchvision
    else
        echo "Warning: Requirements file not found at $REQUIREMENTS_PATH"
    fi
else
    echo "Using existing conda environment: $CONDA_ENV_NAME"
    conda activate $CONDA_ENV_NAME
fi

# Run the Python script
echo "Running batch-alpha2.py script..."
echo "Copying images from dataset directory to joycaption dir (the devs made a mistake I'm too lazy to fix sorry about this)"
find /dataset_here -type f -exec cp {} $INPUT_DIR \;
python $SCRIPT_PATH
echo "captioning complete"
find $INPUT_DIR -type f -exec cp {} /dataset_here \;

echo "Script execution completed successfully."
echo "The conda environment '$CONDA_ENV_NAME' is preserved for future use."
