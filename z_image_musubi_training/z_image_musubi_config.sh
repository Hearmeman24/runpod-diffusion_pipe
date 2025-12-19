# The file extension is purely so it shows up with nice colors on Jupyter, only god can judge me for making stupid decisions.

# ====== Z Image (Turbo) Musubi Config File ======

# Optional override (default in setup script):
# WORKDIR="$NETWORK_VOLUME/z_image_musubi_training"

# ---- Dataset ----
# Where your images + captions live.
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"

# Optional caption extension
CAPTION_EXT=".txt"

# Dataset repeats
NUM_REPEATS=10

# Z Image recipe uses batch_size=1 by default
BATCH_SIZE=1

# resolution for dataset.toml
# e.g. "1024, 1024"
RESOLUTION_LIST="1024, 1024"

# ---- Output ----
OUTPUT_DIR="./outputs"
OUTPUT_NAME="my_z_image_lora"

# ---- Training schedule ----
SAVE_EVERY_N_EPOCHS=5
MAX_TRAIN_EPOCHS=40

# ---- LoRA ----
NETWORK_DIM=32
NETWORK_ALPHA=32

# ---- Optimizer (Prodigy) ----
# Prodigy is self-adaptive; it expects lr=1.0.
LEARNING_RATE=1.0
TEXT_ENCODER_LR=1.0

# Optimizer args are passed as separate CLI args.
# shellcheck disable=SC2034
OPTIMIZER_ARGS=(
  "decouple=True"
  "weight_decay=0.01"
  "use_bias_correction=True"
)

# ---- Misc ----
PRIOR_LOSS_WEIGHT=1.0

# Flags to control repeatable behavior
FORCE_SETUP=0
KEEP_DATASET=0
