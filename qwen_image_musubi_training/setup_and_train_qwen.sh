#!/usr/bin/env bash
set -euo pipefail

########################################
# GPU detection
########################################
gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}'
  elif [ -n "${CUDA_VISIBLE_DEVICES-}" ] && [ "${CUDA_VISIBLE_DEVICES-}" != "" ]; then
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

########################################
# Load user config
########################################
CONFIG_FILE="${CONFIG_FILE:-qwen_musubi_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Config file '$CONFIG_FILE' not found. Create it and re-run."
  echo "Tip: use a Bash-y config with syntax highlighting, e.g.: qwen_musubi_config.sh"
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

########################################
# Helpers for numeric CSV -> TOML arrays
########################################
normalize_numeric_csv() {
  # Input: "720, 896, 1152" or "[720, 896, 1152]" or '"720, 896, 1152"'
  # Output: "720, 896, 1152"
  local s="$1"
  s="$(echo "$s" | tr -d '[]"' )"
  # collapse spaces around commas; trim leading/trailing spaces
  s="$(echo "$s" | sed -E 's/[[:space:]]*,[[:space:]]*/, /g; s/^[[:space:]]+|[[:space:]]+$//g')"
  echo "$s"
}

# Normalize lists (with defaults if not set)
RESOLUTION_LIST_NORM="$(normalize_numeric_csv "${RESOLUTION_LIST:-"1024, 1024"}")"

# Basic sanity checks
[[ "$RESOLUTION_LIST_NORM" =~ ^[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*$ ]] || { echo "Bad RESOLUTION_LIST; expected comma-separated ints."; exit 1; }

########################################
# Derived paths (from WORKDIR & DATASET_DIR)
########################################
WORKDIR="${WORKDIR:-$NETWORK_VOLUME/qwen_image_musubi_training}"
DATASET_DIR="${DATASET_DIR:-$WORKDIR/dataset_here}"

# Qwen has its own independent models directory
MODELS_DIR="$WORKDIR/models"

REPO_DIR="$WORKDIR/musubi-tuner"

QWEN_DIT="$MODELS_DIR/diffusion_models/split_files/diffusion_models/qwen_image_bf16.safetensors"
QWEN_VAE="$MODELS_DIR/vae/vae/diffusion_pytorch_model.safetensors"
QWEN_TEXT_ENCODER="$MODELS_DIR/text_encoders/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"

OUTPUT_DIR="$WORKDIR/output"
TITLE="${TITLE:-Qwen_image_lora}"
AUTHOR="${AUTHOR:-HearmemanAI}"

SETUP_MARKER="$REPO_DIR/.setup_done"

# Config-driven knobs (with safe defaults)
LORA_RANK="${LORA_RANK:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-16}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
DATASET_TYPE="${DATASET_TYPE:-image}"

# Image specific defaults if missing
CAPTION_EXT="${CAPTION_EXT:-.txt}"
NUM_REPEATS="${NUM_REPEATS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

# Flags to control repeatable behavior
FORCE_SETUP="${FORCE_SETUP:-0}"
SKIP_CACHE="${SKIP_CACHE:-0}"
KEEP_DATASET="${KEEP_DATASET:-0}"

########################################
# One-time setup (0–4)
########################################
if [ ! -f "$SETUP_MARKER" ] || [ "$FORCE_SETUP" = "1" ]; then
  echo ">>> Running one-time setup (0–4)..."

  # 0) Basic folders
  mkdir -p "$WORKDIR" "$DATASET_DIR"
  mkdir -p "$MODELS_DIR"/{text_encoders,vae,diffusion_models}

  # 1) Clone Musubi (reuse if already exists from Wan2.2)
  cd "$WORKDIR"
  if [ ! -d "$REPO_DIR/.git" ]; then
    echo ">>> Cloning Musubi into $REPO_DIR"
    git clone --recursive https://github.com/kohya-ss/musubi-tuner.git "$REPO_DIR"
  else
    echo ">>> Musubi already present; updating submodules"
    git -C "$REPO_DIR" submodule update --init --recursive
  fi

  # 2a) System deps + venv (create venv one-time)
  apt-get update -y
  apt-get install -y python3-venv
  cd "$REPO_DIR"
  if [ ! -d "venv" ]; then python3 -m venv venv; fi
  source venv/bin/activate

  # 3) Python deps
  pip install -e .
  pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
  pip install protobuf six huggingface_hub==0.34.0
  pip install hf_transfer hf_xet || true
  export HF_HUB_ENABLE_HF_TRANSFER=1 || true

  # 4) Download models (idempotent)
  echo ">>> Downloading Qwen-Image models to $MODELS_DIR ..."
  
  # Create necessary subdirectories for split files
  mkdir -p "$MODELS_DIR/diffusion_models/split_files/diffusion_models"
  mkdir -p "$MODELS_DIR/text_encoders/split_files/text_encoders"
  mkdir -p "$MODELS_DIR/vae/vae"
  
  # Download Qwen-Image DiT
  hf download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_bf16.safetensors \
    --local-dir "$MODELS_DIR/diffusion_models"
  
  # Download Qwen Text Encoder (Qwen2.5-VL)
  hf download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
    --local-dir "$MODELS_DIR/text_encoders"
  
  # Download Qwen VAE
  hf download Qwen/Qwen-Image vae/diffusion_pytorch_model.safetensors \
    --local-dir "$MODELS_DIR/vae"

  touch "$SETUP_MARKER"
  echo ">>> Setup complete."
else
  echo ">>> Setup already done (found $SETUP_MARKER). Skipping 0–4."
  cd "$REPO_DIR"
  source venv/bin/activate
fi

########################################
# 5) Create/keep dataset.toml for image training
########################################
mkdir -p "$REPO_DIR/dataset"
DATASET_TOML="$REPO_DIR/dataset/dataset.toml"

if [ "$KEEP_DATASET" = "1" ] && [ -f "$DATASET_TOML" ]; then
  echo ">>> KEEP_DATASET=1 set and dataset.toml exists; leaving it as-is."
else
  echo ">>> Writing dataset.toml for DATASET_TYPE=$DATASET_TYPE"
  cat > "$DATASET_TOML" <<TOML
[general]
resolution = [${RESOLUTION_LIST_NORM}]
caption_extension = "$CAPTION_EXT"
batch_size = ${BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = false
num_repeats = ${NUM_REPEATS}

[[datasets]]
image_directory = "$DATASET_DIR"
cache_directory = "$DATASET_DIR/cache"
num_repeats = ${NUM_REPEATS}
TOML

  echo ">>> dataset.toml written:"
  sed -n '1,200p' "$DATASET_TOML"
fi

########################################
# 6) Cache latents + Text Encoder (skippable)
########################################
if [ "$SKIP_CACHE" = "1" ]; then
  echo ">>> SKIP_CACHE=1 set; skipping latent & Text Encoder caching."
else
  echo ">>> Caching latents..."
  python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config "$DATASET_TOML" \
    --vae "$QWEN_VAE"

  echo ">>> Caching Text Encoder outputs..."
  python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET_TOML" \
    --text_encoder "$QWEN_TEXT_ENCODER" \
    --batch_size 1
fi

########################################
# 7) Training env niceties
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

########################################
# 8) Launch training — built from config
########################################
mkdir -p "$OUTPUT_DIR"
echo ">>> Launching training with:"
echo "    rank=$LORA_RANK, max_epochs=$MAX_EPOCHS, save_every=$SAVE_EVERY, lr=$LEARNING_RATE"

COMMON_FLAGS=(
  --dit "$QWEN_DIT"
  --vae "$QWEN_VAE"
  --text_encoder "$QWEN_TEXT_ENCODER"
  --dataset_config "$DATASET_TOML"
  --sdpa --mixed_precision bf16
  --timestep_sampling shift
  --weighting_scheme none
  --discrete_flow_shift 3.0       # CHANGED: 3.0 is the sweet spot for Qwen flow matching
  --optimizer_type adamw8bit
  --optimizer_args weight_decay=0.01  # CHANGED: Standard decay is sufficient
  --learning_rate "$LEARNING_RATE"           # RECOMMENDATION: Start here. If unstable, drop to 5e-5
  --gradient_checkpointing
  --gradient_accumulation_steps 4 # CRITICAL FIX: Simulates Batch Size 4. Smoothes out the loss.
  --max_data_loader_n_workers 2 --persistent_data_loader_workers
  --network_module networks.lora_qwen_image
  --network_dim "$LORA_RANK"                # CHANGED: 64 captures more style nuance than 32
  --network_alpha "$LORA_RANK"              # CHANGED: Alpha = Dim/2 is standard for stability
  --max_grad_norm 1.0             # CRITICAL FIX: Re-enables safety brakes
  --lr_scheduler cosine           # CHANGED: Cosine is much smoother for style training
  --lr_scheduler_num_cycles 1     # Ensures a smooth curve over the whole run
  --max_train_epochs "$MAX_EPOCHS"
  --save_every_n_epochs "$SAVE_EVERY"
  --seed "$SEED"
  --output_dir "$OUTPUT_DIR"
  --output_name "$TITLE"
  --metadata_title "$TITLE"
  --metadata_author "$AUTHOR"
)

echo ">>> Starting Qwen-Image training..."
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/qwen_image_train_network.py \
  "${COMMON_FLAGS[@]}"

echo ">>> Done."

