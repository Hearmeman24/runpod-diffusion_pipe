#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for better UX
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${WHITE}$1${NC}"
    echo -e "${CYAN}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Welcome message
clear
print_header "Welcome to HearmemanAI LoRA Trainer using Diffusion Pipe"
echo ""
echo -e "${PURPLE}This interactive script will guide you through setting up and starting a LoRA training session.${NC}"
echo ""

# Model selection
echo -e "${WHITE}Please select the model you want to train:${NC}"
echo ""
echo "1) Flux"
echo "2) SDXL"
echo "3) Wan 1.3B"
echo "4) Wan 14B Text-To-Video (Supports both T2V and I2V)"
echo "5) Wan 14B Image-To-Video (Not recommended, for advanced users only)"
echo ""

while true; do
    read -p "Enter your choice (1-5): " model_choice
    case $model_choice in
        1)
            MODEL_TYPE="flux"
            MODEL_NAME="Flux"
            TOML_FILE="flux.toml"
            break
            ;;
        2)
            MODEL_TYPE="sdxl"
            MODEL_NAME="SDXL"
            TOML_FILE="sdxl.toml"
            break
            ;;
        3)
            MODEL_TYPE="wan13"
            MODEL_NAME="Wan 1.3B"
            TOML_FILE="wan13_video.toml"
            break
            ;;
        4)
            MODEL_TYPE="wan14b_t2v"
            MODEL_NAME="Wan 14B Text-To-Video"
            TOML_FILE="wan14b_t2v.toml"
            break
            ;;
        5)
            MODEL_TYPE="wan14b_i2v"
            MODEL_NAME="Wan 14B Image-To-Video"
            TOML_FILE="wan14b_i2v.toml"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-5."
            ;;
    esac
done

echo ""
print_success "Selected model: $MODEL_NAME"
echo ""

# Check and set required API keys
if [ "$MODEL_TYPE" = "flux" ]; then
    if [ -z "$HUGGING_FACE_TOKEN" ] || [ "$HUGGING_FACE_TOKEN" = "token_here" ]; then
        print_warning "Hugging Face token is required for Flux model."
        echo ""
        echo "You can get your token from: https://huggingface.co/settings/tokens"
        echo ""
        read -p "Please enter your Hugging Face token: " hf_token
        if [ -z "$hf_token" ]; then
            print_error "Token cannot be empty. Exiting."
            exit 1
        fi
        export HUGGING_FACE_TOKEN="$hf_token"
        print_success "Hugging Face token set successfully."
    else
        print_success "Hugging Face token already set."
    fi
fi

echo ""

# Dataset selection
print_header "Dataset Configuration"
echo ""
echo -e "${WHITE}Do you want to caption images and/or videos?${NC}"
echo ""
echo "1) Images only"
echo "2) Videos only"
echo "3) Both images and videos"
echo "4) Skip captioning (use existing captions)"
echo ""

while true; do
    read -p "Enter your choice (1-4): " caption_choice
    case $caption_choice in
        1)
            CAPTION_MODE="images"
            break
            ;;
        2)
            CAPTION_MODE="videos"
            break
            ;;
        3)
            CAPTION_MODE="both"
            break
            ;;
        4)
            CAPTION_MODE="skip"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-4."
            ;;
    esac
done

echo ""

# Check dataset directories
if [ "$CAPTION_MODE" != "skip" ]; then
    IMAGE_DIR="$NETWORK_VOLUME/image_dataset_here"
    VIDEO_DIR="$NETWORK_VOLUME/video_dataset_here"

    # Check Gemini API key if video captioning is needed
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "token_here" ]; then
            print_warning "Gemini API key is required for video captioning."
            echo ""
            echo "You can get your API key from: https://aistudio.google.com/app/apikey"
            echo ""
            read -p "Please enter your Gemini API key: " gemini_key
            if [ -z "$gemini_key" ]; then
                print_error "API key cannot be empty. Exiting."
                exit 1
            fi
            export GEMINI_API_KEY="$gemini_key"
            print_success "Gemini API key set successfully."
        else
            print_success "Gemini API key already set."
        fi
        echo ""
    fi

    # Function to check if directory has files
    check_directory() {
        local dir=$1
        local type=$2

        if [ ! -d "$dir" ]; then
            print_error "$type directory does not exist: $dir"
            return 1
        fi

        # Check for files (not just directories)
        if [ "$type" = "Image" ]; then
            file_count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        else
            file_count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l)
        fi

        if [ "$file_count" -eq 0 ]; then
            print_error "No $type files found in: $dir"
            return 1
        fi

        print_success "Found $file_count $type file(s) in: $dir"
        return 0
    }

    # Check based on caption mode
    case $CAPTION_MODE in
        "images")
            if ! check_directory "$IMAGE_DIR" "Image"; then
                echo ""
                print_error "Please add images to $IMAGE_DIR and re-run this script."
                exit 1
            fi
            ;;
        "videos")
            if ! check_directory "$VIDEO_DIR" "Video"; then
                echo ""
                print_error "Please add videos to $VIDEO_DIR and re-run this script."
                exit 1
            fi
            ;;
        "both")
            images_ok=true
            videos_ok=true

            if ! check_directory "$IMAGE_DIR" "Image"; then
                images_ok=false
            fi

            if ! check_directory "$VIDEO_DIR" "Video"; then
                videos_ok=false
            fi

            if [ "$images_ok" = false ] || [ "$videos_ok" = false ]; then
                echo ""
                print_error "Please add the missing files and re-run this script."
                if [ "$images_ok" = false ]; then
                    echo "  - Add images to: $IMAGE_DIR"
                fi
                if [ "$videos_ok" = false ]; then
                    echo "  - Add videos to: $VIDEO_DIR"
                fi
                exit 1
            fi
            ;;
    esac
fi

echo ""
print_success "Dataset validation completed successfully!"
echo ""

# Summary
print_header "Training Configuration Summary"
echo ""
echo -e "${WHITE}Model:${NC} $MODEL_NAME"
echo -e "${WHITE}TOML Config:${NC} $TOML_FILE"
echo -e "${WHITE}Caption Mode:${NC} $CAPTION_MODE"

if [ "$MODEL_TYPE" = "flux" ]; then
    echo -e "${WHITE}Hugging Face Token:${NC} Set ✓"
fi

if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    echo -e "${WHITE}Gemini API Key:${NC} Set ✓"
fi

echo ""
print_info "Configuration completed! Starting model download and setup..."
echo ""

# Model download logic
print_header "Downloading Required Models"
echo ""

mkdir -p "$NETWORK_VOLUME/models"

case $MODEL_TYPE in
    "flux")
        if [ -z "$HUGGING_FACE_TOKEN" ] || [ "$HUGGING_FACE_TOKEN" == "token_here" ]; then
            print_error "HUGGING_FACE_TOKEN is not set properly."
            exit 1
        fi

        print_info "HUGGING_FACE_TOKEN is set."
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved flux.toml to examples directory"
        fi
        print_info "Downloading Flux model..."
        mkdir -p "$NETWORK_VOLUME/models/flux"
        huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "$NETWORK_VOLUME/models/flux" --repo-type model --token "$HUGGING_FACE_TOKEN" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
        print_success "Finished downloading Flux model"
        ;;

    "sdxl")
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved sdxl.toml to examples directory"
        fi
        print_info "Downloading Base SDXL model..."
        huggingface-cli download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir "$NETWORK_VOLUME/models/" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
        print_success "Finished downloading base SDXL model"
        ;;

    "wan13")
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved wan13_video.toml to examples directory"
        fi
        print_info "Downloading Wan 1.3B model..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B"
        huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
        print_success "Finished downloading Wan 1.3B model"
        ;;

    "wan14b_t2v")
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved wan14b_t2v.toml to examples directory"
        fi
        print_info "Downloading Wan 14B T2V model..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B"
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
        print_success "Finished downloading Wan 14B T2V model"
        ;;

    "wan14b_i2v")
        if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml" ]; then
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved wan14b_i2v.toml to examples directory"
        fi
        print_info "Downloading Wan 14B I2V model..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P"
        huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" 2>&1 | tee "$NETWORK_VOLUME/download_log.txt"
        print_success "Finished downloading Wan 14B I2V model"
        ;;
esac

echo ""
print_success "Model download completed!"
echo ""

# Update dataset.toml file with actual paths and video config
print_header "Configuring Dataset"
echo ""

DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"

if [ -f "$DATASET_TOML" ]; then
    print_info "Updating dataset.toml with actual paths..."

    # Create backup
    cp "$DATASET_TOML" "$DATASET_TOML.backup"

    # Replace $NETWORK_VOLUME with actual path in image directory
    sed -i "s|\$NETWORK_VOLUME/image_dataset_here|$NETWORK_VOLUME/image_dataset_here|g" "$DATASET_TOML"

    # Replace $NETWORK_VOLUME with actual path in video directory (even if commented)
    sed -i "s|\$NETWORK_VOLUME/video_dataset_here|$NETWORK_VOLUME/video_dataset_here|g" "$DATASET_TOML"

    # Uncomment video dataset section if user wants to caption videos
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Enabling video dataset in configuration..."
        # Uncomment the video directory section
        sed -i '/# \[\[directory\]\]/,/# num_repeats = 5/ s/^# //' "$DATASET_TOML"
    fi

    print_success "Dataset configuration updated"
else
    print_warning "dataset.toml not found at $DATASET_TOML"
fi

# Extract and display training configuration summary
print_header "Training Configuration Summary"
echo ""

# Read resolution from dataset.toml
if [ -f "$DATASET_TOML" ]; then
    RESOLUTION=$(grep "^resolutions = " "$DATASET_TOML" | sed 's/resolutions = \[\([0-9]*\)\]/\1/')
    if [ -z "$RESOLUTION" ]; then
        RESOLUTION="1024 (default)"
    fi
else
    RESOLUTION="1024 (default)"
fi

# Read training parameters from model TOML file
MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
if [ -f "$MODEL_TOML" ]; then
    EPOCHS=$(grep "^epochs = " "$MODEL_TOML" | sed 's/epochs = //')
    SAVE_EVERY=$(grep "^save_every_n_epochs = " "$MODEL_TOML" | sed 's/save_every_n_epochs = //')
    RANK=$(grep "^rank = " "$MODEL_TOML" | sed 's/rank = //')
    LR=$(grep "^lr = " "$MODEL_TOML" | sed 's/lr = //')
    OPTIMIZER_TYPE=$(grep "^type = " "$MODEL_TOML" | grep -A5 "\[optimizer\]" | grep "^type = " | sed "s/type = '//;s/'//")

    # Set defaults if not found
    [ -z "$EPOCHS" ] && EPOCHS="1000 (default)"
    [ -z "$SAVE_EVERY" ] && SAVE_EVERY="2 (default)"
    [ -z "$RANK" ] && RANK="32 (default)"
    [ -z "$LR" ] && LR="2e-5 (default)"
    [ -z "$OPTIMIZER_TYPE" ] && OPTIMIZER_TYPE="adamw_optimi (default)"
else
    # Fallback defaults if TOML file not found
    EPOCHS="1000 (default)"
    SAVE_EVERY="2 (default)"
    RANK="32 (default)"
    LR="2e-5 (default)"
    OPTIMIZER_TYPE="adamw_optimi (default)"
fi

echo -e "${WHITE}Model:${NC} $MODEL_NAME"
echo -e "${WHITE}TOML Config:${NC} examples/$TOML_FILE"
echo -e "${WHITE}Resolution:${NC} ${RESOLUTION}x${RESOLUTION}"
echo ""

echo -e "${WHITE}Training Parameters:${NC}"
echo "  📊 Epochs: $EPOCHS"
echo "  💾 Save Every: $SAVE_EVERY epochs"
echo "  🎛️  LoRA Rank: $RANK"
echo "  📈 Learning Rate: $LR"
echo "  ⚙️  Optimizer: $OPTIMIZER_TYPE"
echo ""

# Show dataset paths and repeats
if [ "$CAPTION_MODE" != "skip" ]; then
    echo -e "${WHITE}Dataset Configuration:${NC}"

    # Always show image dataset info
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        IMAGE_COUNT=$(find "$NETWORK_VOLUME/image_dataset_here" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.gif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
        echo "  📷 Images: $NETWORK_VOLUME/image_dataset_here ($IMAGE_COUNT files)"
        echo "     Repeats: 1 per epoch"
    fi

    # Show video dataset info if applicable
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        VIDEO_COUNT=$(find "$NETWORK_VOLUME/video_dataset_here" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l)
        echo "  🎬 Videos: $NETWORK_VOLUME/video_dataset_here ($VIDEO_COUNT files)"
        echo "     Repeats: 5 per epoch"
    fi
else
    echo -e "${WHITE}Dataset:${NC} Using existing captions"
fi

if [ "$MODEL_TYPE" = "flux" ]; then
    echo -e "${WHITE}Hugging Face Token:${NC} Set ✓"
fi

if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    echo -e "${WHITE}Gemini API Key:${NC} Set ✓"
fi

echo ""

# Prompt user about configuration files
print_header "Training Configuration"
echo ""

print_info "Before starting training, you can modify the default training parameters in these files:"
echo ""
echo -e "${WHITE}1. Model Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
echo ""
echo -e "${WHITE}2. Dataset Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
echo ""

print_warning "These files contain important settings like:"
echo "  • Learning rate, batch size, epochs"
echo "  • Dataset paths and image/video resolutions"
echo "  • LoRA rank and other adapter settings"
echo ""

echo -e "${YELLOW}Would you like to modify these files before starting training?${NC}"
echo "1) Continue with default settings"
echo "2) Pause here - I'll modify the files manually"
echo ""

while true; do
    read -p "Enter your choice (1-2): " config_choice
    case $config_choice in
        1)
            print_success "Continuing with default training settings..."
            break
            ;;
        2)
            print_info "Training paused. Please modify the configuration files as needed."
            echo ""
            echo -e "${WHITE}When ready to start training, run:${NC}"
            echo "cd $NETWORK_VOLUME/diffusion_pipe"
            echo "NCCL_P2P_DISABLE=\"1\" NCCL_IB_DISABLE=\"1\" deepspeed --num_gpus=1 train.py --deepspeed --config examples/$TOML_FILE"
            echo ""
            print_success "Script completed. Modify your files and run the command above when ready."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please enter 1 or 2."
            ;;
    esac
done

echo ""

# Start training
print_header "Starting Training"
echo ""

print_info "Changing to diffusion_pipe directory..."
cd "$NETWORK_VOLUME/diffusion_pipe"

print_info "Starting LoRA training with $MODEL_NAME..."
print_info "Using configuration: examples/$TOML_FILE"
echo ""

print_warning "Training is starting. This may take several hours depending on your dataset size and model."
print_info "You can monitor progress in the console output below."
echo ""

# Start training with the appropriate TOML file
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config "examples/$TOML_FILE"

print_success "Training completed!"