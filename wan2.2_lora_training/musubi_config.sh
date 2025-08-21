# The file extension is purely so it shows up with nice colors on Jupyter, only god can judge me for making stupid decisions.

# ====== Wan 2.2 Config File ======
# LoRA rank drives both network_dim and network_alpha
LORA_RANK=16

# training schedule
MAX_EPOCHS=100
SAVE_EVERY=20

# seeds (used for the two jobs; if only 1 GPU you'll be asked to pick high/low)
SEED_HIGH=41
SEED_LOW=42

# optimizer
LEARNING_RATE=3e-4

# dataset: "video" or "image"
DATASET_TYPE=image

# resolution list for bucketed training (must be TOML-ish array)
# e.g. [896, 1152] or [1024, 1024]
RESOLUTION_LIST="1024, 1024"

# common dataset paths (adjust if you keep data elsewhere)
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"

# Select LoRA Names
TITLE_HIGH="Your High Noise LoRA Name Here"
TITLE_LOW="Your Low Noise LoRA Name Here"

# ---- IMAGE options (used only when DATASET_TYPE=image) ----
BATCH_SIZE=1
NUM_REPEATS=1

# ---- VIDEO options (used only when DATASET_TYPE=video) ----
# frames per sample; TOML array (Musubi rounds like [1,57,117])
TARGET_FRAMES="1, 57, 117"
FRAME_EXTRACTION="head"     # head | middle | tail (per wan2.2_lora_training docs)
NUM_REPEATS=1

# Optional caption extension used by both modes
CAPTION_EXT=".txt"
