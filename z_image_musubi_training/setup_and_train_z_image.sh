#!/usr/bin/env bash
set -euo pipefail

if [ -z "${NETWORK_VOLUME-}" ]; then
  echo "ERROR: NETWORK_VOLUME is not set. Run this inside the RunPod template environment (or export NETWORK_VOLUME)."
  exit 1
fi

########################################
# GPU detection
########################################
gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}'
  elif [ -n "${CUDA_VISIBLE_DEVICES-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then
    echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}'
  else
    echo 0
  fi
}
GPU_COUNT=$(gpu_count)
echo ">>> Detected GPUs: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "ERROR: No CUDA GPUs detected. Aborting."
  exit 1
fi

########################################
# CUDA compatibility check
########################################
check_cuda_compatibility() {
    python3 << 'PYTHON_EOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
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

########################################
# Load user config
########################################
CONFIG_FILE="${CONFIG_FILE:-z_image_musubi_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Config file '$CONFIG_FILE' not found. Create it and re-run."
  echo "Tip: use a Bash-y config with syntax highlighting, e.g.: z_image_musubi_config.sh"
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

########################################
# Helpers for numeric CSV -> TOML arrays
########################################
normalize_numeric_csv() {
  # Input: "1024, 1024" or "[1024, 1024]" or '"1024, 1024"'
  # Output: "1024, 1024"
  local s="$1"
  s="$(echo "$s" | tr -d '[]"' )"
  s="$(echo "$s" | sed -E 's/[[:space:]]*,[[:space:]]*/, /g; s/^[[:space:]]+|[[:space:]]+$//g')"
  echo "$s"
}

RESOLUTION_LIST_NORM="$(normalize_numeric_csv "${RESOLUTION_LIST:-"1024, 1024"}")"
[[ "$RESOLUTION_LIST_NORM" =~ ^[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*$ ]] || { echo "Bad RESOLUTION_LIST; expected comma-separated ints."; exit 1; }

########################################
# Derived paths (from WORKDIR & DATASET_DIR)
########################################
WORKDIR="${WORKDIR:-$NETWORK_VOLUME/z_image_musubi_training}"
DATASET_DIR="${DATASET_DIR:-$NETWORK_VOLUME/image_dataset_here}"

REPO_DIR="$WORKDIR/musubi-tuner"

# Shared Z Image weights (same locations as interactive_start_training.sh)
MODELS_DIR="$NETWORK_VOLUME/models/z_image"
ZIMAGE_MODEL="$MODELS_DIR/z_image_turbo_bf16.safetensors"
ZIMAGE_VAE="$MODELS_DIR/ae.safetensors"
ZIMAGE_TEXT_ENCODER="$MODELS_DIR/qwen_3_4b.safetensors"
ZIMAGE_BASE_WEIGHTS="$MODELS_DIR/zimage_turbo_training_adapter_v2.safetensors"

# Output dir: allow relative paths (default "./outputs")
OUTPUT_DIR_RAW="${OUTPUT_DIR:-./outputs}"
if [[ "$OUTPUT_DIR_RAW" == /* ]]; then
  OUTPUT_DIR_FINAL="$OUTPUT_DIR_RAW"
else
  # If user passes ./outputs, keep it relative to the repo directory.
  OUTPUT_DIR_FINAL="$REPO_DIR/${OUTPUT_DIR_RAW#./}"
fi
OUTPUT_NAME_FINAL="${OUTPUT_NAME:-my_z_image_lora}"

DATASET_TOML_DIR="$REPO_DIR/dataset"
DATASET_TOML="$DATASET_TOML_DIR/dataset.toml"

SETUP_MARKER="$REPO_DIR/.setup_done"

# Config-driven knobs (safe defaults)
CAPTION_EXT="${CAPTION_EXT:-.txt}"
NUM_REPEATS="${NUM_REPEATS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"

SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-5}"
MAX_TRAIN_EPOCHS="${MAX_TRAIN_EPOCHS:-40}"

NETWORK_DIM="${NETWORK_DIM:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-32}"

LEARNING_RATE="${LEARNING_RATE:-1.0}"
TEXT_ENCODER_LR="${TEXT_ENCODER_LR:-1.0}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"

FORCE_SETUP="${FORCE_SETUP:-0}"
KEEP_DATASET="${KEEP_DATASET:-0}"

########################################
# One-time setup (0–3)
########################################
if [ ! -f "$SETUP_MARKER" ] || [ "$FORCE_SETUP" = "1" ]; then
  echo ">>> Running one-time setup (0–3)..."

  # 0) Basic folders
  mkdir -p "$WORKDIR" "$MODELS_DIR"

  # 1) Clone Musubi
  cd "$WORKDIR"
  if [ ! -d "$REPO_DIR/.git" ]; then
    echo ">>> Cloning Musubi into $REPO_DIR"
    git clone --recursive https://github.com/kohya-ss/musubi-tuner.git "$REPO_DIR"
  else
    echo ">>> Musubi already present; updating submodules"
    git -C "$REPO_DIR" submodule update --init --recursive
  fi

  # 2) System deps + venv
  apt-get update -y
  apt-get install -y python3-venv
  cd "$REPO_DIR"
  if [ ! -d "venv" ]; then python3 -m venv venv; fi
  # shellcheck disable=SC1091
  source venv/bin/activate

  # 3) Python deps
  pip install -e .
  pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
  pip install protobuf six huggingface_hub==0.34.0
  pip install hf_transfer hf_xet || true
  export HF_HUB_ENABLE_HF_TRANSFER=1 || true

  # Prodigy optimizer package for --optimizer_type prodigyopt.Prodigy
  pip install prodigyopt

  touch "$SETUP_MARKER"
  echo ">>> Setup complete."
else
  echo ">>> Setup already done (found $SETUP_MARKER). Skipping 0–3."
  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

########################################
# Ensure Prodigy optimizer is installed
########################################
if python -c "import prodigyopt" >/dev/null 2>&1; then
  echo ">>> Prodigy optimizer (prodigyopt) already installed."
else
  echo ">>> Installing Prodigy optimizer (prodigyopt)..."
  pip install prodigyopt
fi

########################################
# 4) Ensure Z Image weights exist (idempotent)
########################################
missing_any=0
if [ ! -f "$ZIMAGE_MODEL" ]; then missing_any=1; fi
if [ ! -f "$ZIMAGE_VAE" ]; then missing_any=1; fi
if [ ! -f "$ZIMAGE_TEXT_ENCODER" ]; then missing_any=1; fi

if [ "$missing_any" = "1" ]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 'hf' CLI is not available, but required to download Z Image weights."
    exit 1
  fi

  echo ">>> Missing Z Image core weights; downloading via HuggingFace..."

  TEMP_DIR="$NETWORK_VOLUME/models/z_image_turbo_temp"
  rm -rf "$TEMP_DIR" || true
  mkdir -p "$TEMP_DIR"

  hf download Comfy-Org/z_image_turbo --local-dir "$TEMP_DIR"

  echo ">>> Moving model files to final location: $MODELS_DIR"
  mkdir -p "$MODELS_DIR"

  mv "$TEMP_DIR/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$MODELS_DIR/"
  mv "$TEMP_DIR/split_files/vae/ae.safetensors" "$MODELS_DIR/"
  mv "$TEMP_DIR/split_files/text_encoders/qwen_3_4b.safetensors" "$MODELS_DIR/"

  rm -rf "$TEMP_DIR"
  echo ">>> Z Image core weights ready."
else
  echo ">>> Z Image core weights already present."
fi

if [ ! -f "$ZIMAGE_BASE_WEIGHTS" ]; then
  echo ">>> Downloading Z Image Turbo training adapter..."
  wget -q --show-progress -O "$ZIMAGE_BASE_WEIGHTS" \
    "https://huggingface.co/ostris/zimage_turbo_training_adapter/resolve/main/zimage_turbo_training_adapter_v2.safetensors"
else
  echo ">>> Z Image Turbo training adapter already present."
fi

########################################
# 5) Create/keep dataset.toml
########################################
mkdir -p "$DATASET_TOML_DIR"

if [ "$KEEP_DATASET" = "1" ] && [ -f "$DATASET_TOML" ]; then
  echo ">>> KEEP_DATASET=1 set and dataset.toml exists; leaving it as-is."
else
  echo ">>> Writing dataset.toml for Z Image"
  cat > "$DATASET_TOML" <<TOML
[[datasets]]
resolution = [${RESOLUTION_LIST_NORM}]
batch_size = ${BATCH_SIZE}

[[datasets.subsets]]
image_dir = '${DATASET_DIR}'
caption_extension = '${CAPTION_EXT}'
num_repeats = ${NUM_REPEATS}
TOML

  echo ">>> dataset.toml written:"
  sed -n '1,200p' "$DATASET_TOML"
fi

########################################
# 6) Training env niceties
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

########################################
# 7) Launch training (built from config)
########################################
mkdir -p "$OUTPUT_DIR_FINAL"

echo ">>> Launching Z Image training with:"
echo "    output_dir=$OUTPUT_DIR_FINAL"
echo "    output_name=$OUTPUT_NAME_FINAL"
echo "    epochs=$MAX_TRAIN_EPOCHS, save_every=$SAVE_EVERY_N_EPOCHS"
echo "    network_dim=$NETWORK_DIM, network_alpha=$NETWORK_ALPHA"
echo "    optimizer=prodigyopt.Prodigy (lr=$LEARNING_RATE, text_encoder_lr=$TEXT_ENCODER_LR)"

COMMON_FLAGS=(
  --pretrained_model_name_or_path "$ZIMAGE_MODEL"
  --vae "$ZIMAGE_VAE"
  --text_encoder "$ZIMAGE_TEXT_ENCODER"
  --base_weights "$ZIMAGE_BASE_WEIGHTS"
  --dataset_config "$DATASET_TOML"
  --output_dir "$OUTPUT_DIR_FINAL"
  --output_name "$OUTPUT_NAME_FINAL"
  --save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"
  --max_train_epochs "$MAX_TRAIN_EPOCHS"
  --sdpa
  --mixed_precision bf16
  --network_module networks.lora
  --network_dim "$NETWORK_DIM"
  --network_alpha "$NETWORK_ALPHA"
  --optimizer_type prodigyopt.Prodigy
  --learning_rate "$LEARNING_RATE"
  --text_encoder_lr "$TEXT_ENCODER_LR"
  --lr_scheduler constant
  --gradient_checkpointing
  --prior_loss_weight "$PRIOR_LOSS_WEIGHT"
)

# Add optimizer args if configured as an array
if declare -p OPTIMIZER_ARGS >/dev/null 2>&1; then
  # shellcheck disable=SC2154
  if [ "${#OPTIMIZER_ARGS[@]}" -gt 0 ]; then
    COMMON_FLAGS+=( --optimizer_args "${OPTIMIZER_ARGS[@]}" )
  fi
fi

cd "$REPO_DIR"

echo ">>> Starting Z Image training..."
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/zimage_train_network.py \
  "${COMMON_FLAGS[@]}"

echo ">>> Done."
