# The file extension is purely so it shows up with nice colors on Jupyter, only god can judge me for making stupid decisions.

# ====== Wan 2.2 Config File ======
# LoRA rank drives both network_dim and network_alpha
LORA_RANK=32

# training schedule
MAX_EPOCHS=150
SAVE_EVERY=25

# seeds (used for the two jobs; if only 1 GPU you'll be asked to pick high/low)
SEED_HIGH=41
SEED_LOW=42

# optimizer
LEARNING_RATE=1e-4

# dataset: "video" or "image"
DATASET_TYPE=image

# resolution list for bucketed training (must be TOML-ish array)
# e.g. [896, 1152] or [1024, 1024]
RESOLUTION_LIST="896, 1152"

# common dataset paths (adjust if you keep data elsewhere)
WORKDIR="/musubi"
DATASET_DIR="$WORKDIR/dataset_here"

# ---- VIDEO options (used only when DATASET_TYPE=video) ----
# frames per sample; TOML array (Musubi rounds like [1,57,117])
TARGET_FRAMES="1, 57, 117"
FRAME_EXTRACTION="head"     # head | middle | tail (per musubi docs)
NUM_REPEATS=1

# ---- IMAGE options (used only when DATASET_TYPE=image) ----
BATCH_SIZE=1
NUM_REPEATS=1

# Optional caption extension used by both modes
CAPTION_EXT=".txt"
