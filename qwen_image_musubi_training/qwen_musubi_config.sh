# The file extension is purely so it shows up with nice colors on Jupyter, only god can judge me for making stupid decisions.

# ====== Qwen-Image Musubi Config File ======
# LoRA rank drives both network_dim and network_alpha
LORA_RANK=16

# training schedule
MAX_EPOCHS=16
SAVE_EVERY=1

# seed
SEED=42

# optimizer
LEARNING_RATE=5e-5

# dataset: "image" only for Qwen-Image
DATASET_TYPE=image

# resolution list for bucketed training (must be TOML-ish array)
# e.g. [1024, 1024] or [896, 1152]
RESOLUTION_LIST="1024, 1024"

# common dataset paths (adjust if you keep data elsewhere)
DATASET_DIR="$NETWORK_VOLUME/image_dataset_here"

# Select LoRA Name
TITLE="Your Qwen LoRA Name Here"

# ---- IMAGE options ----
BATCH_SIZE=1
NUM_REPEATS=1

# Optional caption extension
CAPTION_EXT=".txt"



