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
CONFIG_FILE="${CONFIG_FILE:-musubi_config.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Config file '$CONFIG_FILE' not found. Create it and re-run."
  echo "Tip: use a Bash-y config with syntax highlighting, e.g.: musubi_config.sh"
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
RESOLUTION_LIST_NORM="$(normalize_numeric_csv "${RESOLUTION_LIST:-"720, 896, 1152"}")"
TARGET_FRAMES_NORM="$(normalize_numeric_csv "${TARGET_FRAMES:-"1, 57, 117"}")"

# Basic sanity checks
[[ "$RESOLUTION_LIST_NORM" =~ ^[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*$ ]] || { echo "Bad RESOLUTION_LIST; expected comma-separated ints."; exit 1; }
if [[ "${DATASET_TYPE:-video}" == "video" ]]; then
  [[ "$TARGET_FRAMES_NORM" =~ ^[0-9]+([[:space:]]*,[[:space:]]*[0-9]+)*$ ]] || { echo "Bad TARGET_FRAMES; expected comma-separated ints."; exit 1; }
fi

########################################
# Derived paths (from WORKDIR & DATASET_DIR)
########################################
WORKDIR="${WORKDIR:-$NETWORK_VOLUME/wan2.2_lora_training}"
DATASET_DIR="${DATASET_DIR:-$WORKDIR/dataset_here}"

REPO_DIR="$WORKDIR/musubi-tuner"
MODELS_DIR="$WORKDIR/models"

WAN_VAE="$MODELS_DIR/vae/split_files/vae/wan_2.1_vae.safetensors"
WAN_T5="$MODELS_DIR/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
WAN_DIT_HIGH="$MODELS_DIR/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
WAN_DIT_LOW="$MODELS_DIR/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"

OUT_HIGH="$WORKDIR/output/high"
OUT_LOW="$WORKDIR/output/low"
TITLE_HIGH="${TITLE_HIGH:-Wan2.2_model_lora}"
TITLE_LOW="${TITLE_LOW:-Wan2.2_model_lora}"
AUTHOR="${AUTHOR:-HearmemanAI}"

SETUP_MARKER="$REPO_DIR/.setup_done"

# Config-driven knobs (with safe defaults)
LORA_RANK="${LORA_RANK:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
SAVE_EVERY="${SAVE_EVERY:-25}"
SEED_HIGH="${SEED_HIGH:-41}"
SEED_LOW="${SEED_LOW:-42}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
DATASET_TYPE="${DATASET_TYPE:-video}"

# Video/image specific defaults if missing
CAPTION_EXT="${CAPTION_EXT:-.txt}"
NUM_REPEATS="${NUM_REPEATS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

FRAME_EXTRACTION="${FRAME_EXTRACTION:-head}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
FRAME_SAMPLE="${FRAME_SAMPLE:-1}"
MAX_FRAMES="${MAX_FRAMES:-129}"
FP_LATENT_WINDOW_SIZE="${FP_LATENT_WINDOW_SIZE:-9}"

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

  # 1) Clone Musubi
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
  echo ">>> Downloading models to $MODELS_DIR ..."
  hf download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir "$MODELS_DIR/text_encoders"
  hf download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors \
    --local-dir "$MODELS_DIR/vae"
  hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
    --local-dir "$MODELS_DIR/diffusion_models"
  hf download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --local-dir "$MODELS_DIR/diffusion_models"

  touch "$SETUP_MARKER"
  echo ">>> Setup complete."
else
  echo ">>> Setup already done (found $SETUP_MARKER). Skipping 0–4."
  cd "$REPO_DIR"
  source venv/bin/activate
fi

########################################
# 5) Create/keep dataset.toml based on DATASET_TYPE
########################################
mkdir -p "$REPO_DIR/dataset"
DATASET_TOML="$REPO_DIR/dataset/dataset.toml"

if [ "$KEEP_DATASET" = "1" ] && [ -f "$DATASET_TOML" ]; then
  echo ">>> KEEP_DATASET=1 set and dataset.toml exists; leaving it as-is."
else
  echo ">>> Writing dataset.toml for DATASET_TYPE=$DATASET_TYPE"
  if [ "$DATASET_TYPE" = "video" ]; then
    cat > "$DATASET_TOML" <<TOML
[general]
resolution = [${RESOLUTION_LIST_NORM}]
enable_bucket = true
bucket_no_upscale = false
caption_extension = "$CAPTION_EXT"

[[datasets]]
video_directory = "$DATASET_DIR"
target_frames = [${TARGET_FRAMES_NORM}]
frame_extraction = "$FRAME_EXTRACTION"
frame_stride = ${FRAME_STRIDE}
frame_sample = ${FRAME_SAMPLE}
max_frames = ${MAX_FRAMES}
fp_latent_window_size = ${FP_LATENT_WINDOW_SIZE}
batch_size = 1
num_repeats = ${NUM_REPEATS}
TOML
  else
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
  fi

  echo ">>> dataset.toml written:"
  sed -n '1,200p' "$DATASET_TOML"
fi

########################################
# 6) Cache latents + T5 (skippable)
########################################
if [ "$SKIP_CACHE" = "1" ]; then
  echo ">>> SKIP_CACHE=1 set; skipping latent & T5 caching."
else
  python wan_cache_latents.py \
    --dataset_config "$DATASET_TOML" \
    --vae "$WAN_VAE"

  python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET_TOML" \
    --t5 "$WAN_T5"
fi

########################################
# 7) Training env niceties
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

########################################
# ANSI colors for output
########################################
C_RESET="\033[0m"
C_HIGH="\033[38;5;40m"   # green
C_LOW="\033[38;5;214m"   # orange

########################################
# 8) Launch training(s) — built from config
########################################
mkdir -p "$OUT_HIGH" "$OUT_LOW"
echo ">>> Launching training with:"
echo "    rank=$LORA_RANK, max_epochs=$MAX_EPOCHS, save_every=$SAVE_EVERY, lr=$LEARNING_RATE"

COMMON_FLAGS=(
  --task t2v-A14B
  --vae "$WAN_VAE"
  --t5 "$WAN_T5"
  --dataset_config "$DATASET_TOML"
  --xformers --mixed_precision fp16 --fp8_base
  --optimizer_type adamw --optimizer_args weight_decay=0.1
  --learning_rate "$LEARNING_RATE"
  --gradient_checkpointing --gradient_accumulation_steps 1
  --max_data_loader_n_workers 2
  --network_module networks.lora_wan --network_dim "$LORA_RANK" --network_alpha "$LORA_RANK"
  --timestep_sampling shift --discrete_flow_shift 1.0
  --max_grad_norm 0
  --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5"
  --max_train_epochs "$MAX_EPOCHS" --save_every_n_epochs "$SAVE_EVERY"
)

if [ "${GPU_COUNT}" -ge 2 ]; then
  echo ">>> 2+ GPUs: running HIGH on GPU0 and LOW on GPU1"

  echo -e "${C_HIGH}[HIGH] Starting training...${C_RESET}"
  env CUDA_VISIBLE_DEVICES=0 accelerate launch --num_cpu_threads_per_process 8 \
    "$REPO_DIR/wan_train_network.py" \
    --dit "$WAN_DIT_HIGH" \
    --preserve_distribution_shape --min_timestep 875 --max_timestep 1000 \
    --seed "$SEED_HIGH" \
    --output_dir "$OUT_HIGH" \
    --output_name "$TITLE_HIGH" \
    --metadata_title "$TITLE_HIGH" \
    --metadata_author "$AUTHOR" \
    "${COMMON_FLAGS[@]}" &

  echo -e "${C_LOW}[LOW] Starting training...${C_RESET}"
  env CUDA_VISIBLE_DEVICES=1 accelerate launch --num_cpu_threads_per_process 8 \
    "$REPO_DIR/wan_train_network.py" \
    --dit "$WAN_DIT_LOW" \
    --preserve_distribution_shape --min_timestep 0 --max_timestep 875 \
    --seed "$SEED_LOW" \
    --output_dir "$OUT_LOW" \
    --output_name "$TITLE_LOW" \
    --metadata_title "$TITLE_LOW" \
    --metadata_author "$AUTHOR" \
    "${COMMON_FLAGS[@]}" &

  wait
  echo ">>> Both training processes completed."

else
  echo ">>> Only 1 GPU detected."
  echo "Select which model to train:"
  echo "1) HIGH-noise model"
  echo "2) LOW-noise model"
  read -rp "Enter 1 or 2: " choice

  if [ "$choice" = "1" ]; then
    echo ">>> Training HIGH-noise model on GPU0"
    accelerate launch --num_cpu_threads_per_process 8 \
      "$REPO_DIR/wan_train_network.py" \
      --dit "$WAN_DIT_HIGH" \
      --preserve_distribution_shape --min_timestep 875 --max_timestep 1000 \
      --seed "$SEED_HIGH" \
      --output_dir "$OUT_HIGH" \
      --output_name "$TITLE_HIGH" \
      --metadata_title "$TITLE_HIGH" \
      --metadata_author "$AUTHOR" \
      "${COMMON_FLAGS[@]}"
  elif [ "$choice" = "2" ]; then
    echo ">>> Training LOW-noise model on GPU0"
    accelerate launch --num_cpu_threads_per_process 8 \
      "$REPO_DIR/wan_train_network.py" \
      --dit "$WAN_DIT_LOW" \
      --preserve_distribution_shape --min_timestep 0 --max_timestep 875 \
      --seed "$SEED_LOW" \
      --output_dir "$OUT_LOW" \
      --output_name "$TITLE_LOW" \
      --metadata_title "$TITLE_LOW" \
      --metadata_author "$AUTHOR" \
      "${COMMON_FLAGS[@]}"
  else
    echo "Invalid choice. Aborting."
    exit 1
  fi
fi

echo ">>> Done."