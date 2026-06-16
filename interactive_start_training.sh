#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Global error handler — catches unexpected failures from set -e
trap 'on_error $LINENO' ERR

on_error() {
    local line_number=$1
    echo ""
    echo -e "\033[0;31m================================================\033[0m"
    echo -e "\033[0;31m  ERROR: Script failed unexpectedly at line $line_number\033[0m"
    echo -e "\033[0;31m================================================\033[0m"
    echo ""
    echo -e "\033[0;34m  What happened:\033[0m"
    echo -e "    A command exited with an error that the script didn't expect."
    echo ""
    echo -e "\033[0;34m  What to check:\033[0m"
    echo -e "    1. Review the output above this message for the actual error"
    echo -e "    2. Check logs in: \$NETWORK_VOLUME/logs/"
    echo -e "       - model_download.log    (model download issues)"
    echo -e "       - image_captioning.log  (captioning issues)"
    echo -e "       - video_captioning.log  (video captioning issues)"
    echo -e "    3. Make sure your datasets are in the correct directories"
    echo -e "    4. Verify you have enough disk space: df -h"
    echo ""
    echo -e "\033[0;34m  If the issue persists:\033[0m"
    echo -e "    - Try restarting the pod and running the script again"
    echo -e "    - Check GitHub issues for known problems"
    echo ""
}

# Colors for better UX - compatible with both light and dark terminals
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
WHITE='\033[0;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${BOLD}$1${NC}"
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
echo -e "${RED}Before you start, make sure to add your datasets to their respective folders.${NC}"
echo ""

# Check for Blackwell GPU and warn user
if [ -f /tmp/gpu_arch_type ]; then
    GPU_ARCH_TYPE=$(cat /tmp/gpu_arch_type)
    DETECTED_GPU=$(cat /tmp/detected_gpu 2>/dev/null || echo "Unknown")
    if [ "$GPU_ARCH_TYPE" = "blackwell" ]; then
        echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}${RED}⚠️  WARNING: BLACKWELL GPU DETECTED ⚠️${NC}"
        echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}${RED}Detected GPU: $DETECTED_GPU${NC}"
        echo -e "${BOLD}${RED}${NC}"
        echo -e "${BOLD}${RED}Blackwell GPUs (B100, B200, RTX 5090, etc.) are very new and${NC}"
        echo -e "${BOLD}${RED}may not be fully supported by all ML libraries yet.${NC}"
        echo -e "${BOLD}${RED}${NC}"
        echo -e "${BOLD}${RED}For best compatibility, use H100 or H200 GPUs.${NC}"
        echo -e "${BOLD}${RED}════════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -n "Continuing in "
        for i in 10 9 8 7 6 5 4 3 2 1; do
            echo -n "$i.."
            sleep 1
        done
        echo ""
        echo ""
    fi
fi

# Create logs directory
mkdir -p "$NETWORK_VOLUME/logs"

# Check if flash-attn installation is still running
# Skip check if wheel was successfully installed in foreground
if [ -f /tmp/flash_attn_wheel_success ]; then
    print_success "flash-attn is installed and ready (installed from prebuilt wheel)."
    echo ""
elif [ -f /tmp/flash_attn_pid ]; then
    FLASH_ATTN_PID=$(cat /tmp/flash_attn_pid)
    if kill -0 "$FLASH_ATTN_PID" 2>/dev/null; then
        print_warning "flash-attn is still being compiled from source (PID: $FLASH_ATTN_PID)"
        print_info "Waiting for flash-attn compilation to complete..."
        print_info "To monitor progress: tail -f $NETWORK_VOLUME/logs/flash_attn_install.log"
        echo ""
        while kill -0 "$FLASH_ATTN_PID" 2>/dev/null; do
            echo -n "."
            sleep 2
        done
        echo ""
        # Check if installation succeeded
        wait "$FLASH_ATTN_PID" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "flash-attn compilation completed successfully!"
        else
            print_warning "flash-attn compilation may have failed. Check log: $NETWORK_VOLUME/logs/flash_attn_install.log"
        fi
        rm -f /tmp/flash_attn_pid
        echo ""
    else
        # Process finished, clean up PID file
        rm -f /tmp/flash_attn_pid
        print_success "flash-attn is installed and ready."
        echo ""
    fi
fi

# Model selection
echo -e "${BOLD}Please select the model you want to train:${NC}"
echo ""
echo "1) Flux"
echo "2) SDXL"
echo "3) Wan 1.3B"
echo "4) Wan 14B Text-To-Video (Supports both T2V and I2V)"
echo "5) Wan 14B Image-To-Video (Not recommended, for advanced users only)"
echo "6) Qwen Image"
echo "7) Z Image Turbo"
echo "8) Qwen Image 2512"
echo ""

while true; do
    read -p "Enter your choice (1-8): " model_choice
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
        6)
            MODEL_TYPE="qwen"
            MODEL_NAME="Qwen Image"
            TOML_FILE="qwen_toml.toml"
            break
            ;;
        7)
            MODEL_TYPE="z_image_turbo"
            MODEL_NAME="Z Image Turbo"
            TOML_FILE="z_image_toml.toml"
            break
            ;;
        8)
            MODEL_TYPE="qwen_2512"
            MODEL_NAME="Qwen Image 2512"
            TOML_FILE="qwen_2512_toml.toml"
            break
            ;;
        *)
            print_error "Invalid choice. Please enter a number between 1-8."
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
            print_error "Token cannot be empty."
            print_info "You can find your token at: https://huggingface.co/settings/tokens"
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
echo -e "${BOLD}Do you want to caption images and/or videos?${NC}"
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
                print_error "API key cannot be empty."
                print_info "Get a Gemini API key at: https://aistudio.google.com/apikey"
                exit 1
            fi
            export GEMINI_API_KEY="$gemini_key"
            print_success "Gemini API key set successfully."
        else
            print_success "Gemini API key already set."
        fi
        echo ""
    fi

    # Ask for trigger word if image captioning is needed
    TRIGGER_WORD=""
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        echo -e "${BOLD}Image Captioning Configuration:${NC}"
        echo ""
        read -p "Enter a trigger word for image captions (or press Enter for none): " TRIGGER_WORD
        if [ -n "$TRIGGER_WORD" ]; then
            print_success "Trigger word set: '$TRIGGER_WORD'"
        else
            print_info "No trigger word set"
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

# CUDA compatibility check
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

print_header "Checking CUDA Compatibility"
check_cuda_compatibility
echo ""

# Model download logic - start in background
print_header "Starting Model Download"
echo ""

mkdir -p "$NETWORK_VOLUME/models"

# Initialize MODEL_DOWNLOAD_PID to ensure it's always set
MODEL_DOWNLOAD_PID=""

case $MODEL_TYPE in
    "flux")
        if [ -z "$HUGGING_FACE_TOKEN" ] || [ "$HUGGING_FACE_TOKEN" = "token_here" ]; then
            print_error "HUGGING_FACE_TOKEN is not set properly."
            print_info "Flux requires a Hugging Face token for download."
            print_info "Set it via the RunPod template environment variables, or enter it when prompted."
            exit 1
        fi

        print_info "HUGGING_FACE_TOKEN is set."
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
        
        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/flux.toml" ]; then
            print_info "flux.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/flux_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/flux.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/flux_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved flux.toml to examples directory"
        else
            print_warning "flux.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/flux.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/flux.toml"
        fi
        print_info "Starting Flux model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/flux"
        hf download black-forest-labs/FLUX.1-dev --local-dir "$NETWORK_VOLUME/models/flux" --repo-type model --token "$HUGGING_FACE_TOKEN" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "sdxl")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
        
        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml" ]; then
            print_info "sdxl.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/sdxl_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/sdxl_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved sdxl.toml to examples directory"
        else
            print_warning "sdxl.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/sdxl.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/sdxl.toml"
        fi
        print_info "Starting Base SDXL model download in background..."
        hf download timoshishi/sdXL_v10VAEFix sdXL_v10VAEFix.safetensors --local-dir "$NETWORK_VOLUME/models/" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "wan13")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
        
        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml" ]; then
            print_info "wan13_video.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/wan13_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/wan13_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved wan13_video.toml to examples directory"
        else
            print_warning "wan13_video.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan13_video.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/wan13_video.toml"
        fi
        print_info "Starting Wan 1.3B model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B"
        hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "wan14b_t2v")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
        
        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml" ]; then
            print_info "wan14b_t2v.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/wan14b_t2v_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/wan14b_t2v_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved wan14b_t2v.toml to examples directory"
        else
            print_warning "wan14b_t2v.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_t2v.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/wan14b_t2v.toml"
        fi
        print_info "Starting Wan 14B T2V model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B"
        hf download Wan-AI/Wan2.1-T2V-14B --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "wan14b_i2v")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
        
        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml" ]; then
            print_info "wan14b_i2v.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/wan14b_i2v_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/wan14b_i2v_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved wan14b_i2v.toml to examples directory"
        else
            print_warning "wan14b_i2v.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/wan14b_i2v.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/wan14b_i2v.toml"
        fi
        print_info "Starting Wan 14B I2V model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P"
        hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "qwen")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
        
        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml" ]; then
            print_info "qwen_toml.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/qwen_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_toml.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/qwen_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_toml.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_toml.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved qwen_toml.toml to examples directory"
        else
            print_warning "qwen_toml.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_toml.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/qwen_toml.toml"
        fi
        print_info "Starting Qwen Image model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Qwen-Image"
        hf download Qwen/Qwen-Image --local-dir "$NETWORK_VOLUME/models/Qwen-Image" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "qwen_2512")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"

        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_2512_toml.toml" ]; then
            print_info "qwen_2512_toml.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/qwen_2512_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/qwen_2512_toml.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_2512_toml.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/qwen_2512_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_2512_toml.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_2512_toml.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved qwen_2512_toml.toml to examples directory"
        else
            print_warning "qwen_2512_toml.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/qwen_2512_toml.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/qwen_2512_toml.toml"
        fi

        # Copy dataset_qwen.toml for 1328 resolution
        if [ ! -f "$NETWORK_VOLUME/diffusion_pipe/examples/dataset_qwen.toml" ]; then
            if [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset_qwen.toml" ]; then
                cp "$NETWORK_VOLUME/runpod-diffusion_pipe/dataset_qwen.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
                print_success "Copied dataset_qwen.toml (1328 resolution) to examples directory"
            else
                print_warning "dataset_qwen.toml not found. Qwen Image 2512 requires 1328 resolution dataset config."
            fi
        else
            print_info "dataset_qwen.toml already exists in examples directory"
        fi

        print_info "Starting Qwen Image 2512 model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/Qwen-Image-2512"
        hf download Qwen/Qwen-Image-2512 --local-dir "$NETWORK_VOLUME/models/Qwen-Image-2512" >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;

    "z_image_turbo")
        # Ensure examples directory exists
        mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"

        # Check if file already exists in destination
        if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml" ]; then
            print_info "z_image_toml.toml already exists in examples directory"
            # Update output_dir even if file already exists
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/z_image_lora'|" "$NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml"
        elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml" ]; then
            # Update output_dir before moving
            sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/z_image_lora'|" "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml"
            mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml" "$NETWORK_VOLUME/diffusion_pipe/examples/"
            print_success "Moved z_image_toml.toml to examples directory"
        else
            print_warning "z_image_toml.toml not found at expected location: $NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml"
            print_warning "Please ensure the file exists or manually copy it to: $NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml"
        fi
        print_info "Starting Z Image Turbo model download in background..."
        mkdir -p "$NETWORK_VOLUME/models/z_image"
        # Download model files using hf download and move to expected location
        (
            echo "Downloading Z Image Turbo models from HuggingFace..."
            # Download main model files (diffusion model, VAE, text encoder)
            hf download Comfy-Org/z_image_turbo --local-dir "$NETWORK_VOLUME/models/z_image_turbo_temp"
            
            echo "Moving model files to final location..."
            # Move files to the expected location
            mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$NETWORK_VOLUME/models/z_image/"
            mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/vae/ae.safetensors" "$NETWORK_VOLUME/models/z_image/"
            mv "$NETWORK_VOLUME/models/z_image_turbo_temp/split_files/text_encoders/qwen_3_4b.safetensors" "$NETWORK_VOLUME/models/z_image/"
            
            # Clean up temp directory
            rm -rf "$NETWORK_VOLUME/models/z_image_turbo_temp"
            
            echo "Downloading Z Image Turbo training adapter..."
            wget -q --show-progress -O "$NETWORK_VOLUME/models/z_image/zimage_turbo_training_adapter_v2.safetensors" \
                "https://huggingface.co/ostris/zimage_turbo_training_adapter/resolve/main/zimage_turbo_training_adapter_v2.safetensors"
            
            echo "Z Image Turbo model download complete!"
        ) >> "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
        MODEL_DOWNLOAD_PID=$!
        ;;
    *)
        print_error "Unknown model type: $MODEL_TYPE"
        print_error "This is a bug in the script — the model selection menu and download handler are out of sync."
        exit 1
        ;;
esac

echo ""

# Start captioning processes if needed
if [ "$CAPTION_MODE" != "skip" ]; then
    print_header "Starting Captioning Process"
    echo ""

    # Clear any existing subfolders in dataset directories before captioning
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Cleaning up image dataset directory..."
        # Remove any subdirectories but keep files
        find "$NETWORK_VOLUME/image_dataset_here" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        print_success "Image dataset directory cleaned"
    fi

    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Cleaning up video dataset directory..."
        # Remove any subdirectories but keep files
        find "$NETWORK_VOLUME/video_dataset_here" -mindepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
        print_success "Video dataset directory cleaned"
    fi

    echo ""

    # Start image captioning in background if needed
    if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Starting image captioning process..."
        JOY_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/JoyCaption/JoyCaptionRunner.sh"

        if [ -f "$JOY_CAPTION_SCRIPT" ]; then
            if [ -n "$TRIGGER_WORD" ]; then
                bash "$JOY_CAPTION_SCRIPT" --trigger-word "$TRIGGER_WORD" > "$NETWORK_VOLUME/logs/image_captioning.log" 2>&1 &
            else
                bash "$JOY_CAPTION_SCRIPT" > "$NETWORK_VOLUME/logs/image_captioning.log" 2>&1 &
            fi
            IMAGE_CAPTION_PID=$!
            print_success "Image captioning started (PID: $IMAGE_CAPTION_PID)"
            print_info "Captioning images..."
            print_info "First run downloads the caption model (~16GB) and may take up to 20 minutes."
            print_info "Subsequent runs will be much faster."
            echo ""

            CAPTION_LOG="$NETWORK_VOLUME/logs/image_captioning.log"
            timeout_counter=0
            max_timeout=3600  # 1 hour timeout
            CAPTION_BAR_WIDTH=30
            SPINNER_CHARS='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
            spinner_idx=0
            progress_started=false

            while kill -0 "$IMAGE_CAPTION_PID" 2>/dev/null; do
                # Check for completion
                if tail -n 1 "$CAPTION_LOG" 2>/dev/null | grep -q "All done!"; then
                    break
                fi
                # Check for errors
                if tail -n 20 "$CAPTION_LOG" 2>/dev/null | grep -qiE "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)"; then
                    echo ""
                    print_error "Image captioning encountered errors. Check log: $CAPTION_LOG"
                    exit 1
                fi

                # Parse progress from "[3/20] Processing ..." or "[3/20] Skipping ..."
                if [ -f "$CAPTION_LOG" ]; then
                    progress_line=$(tail -n 5 "$CAPTION_LOG" 2>/dev/null | grep -oE '\[[0-9]+/[0-9]+\]' | tail -1)
                    if [ -n "$progress_line" ]; then
                        progress_started=true
                        current=$(echo "$progress_line" | grep -oE '[0-9]+' | head -1)
                        total=$(echo "$progress_line" | grep -oE '[0-9]+' | tail -1)
                        if [ -n "$total" ] && [ "$total" -gt 0 ] 2>/dev/null; then
                            pct=$(( current * 100 / total ))
                            filled=$(( pct * CAPTION_BAR_WIDTH / 100 ))
                            empty=$(( CAPTION_BAR_WIDTH - filled ))
                            bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null)
                            bar_empty=$(printf '%0.s░' $(seq 1 $empty 2>/dev/null) 2>/dev/null)
                            printf "\r\033[K  ${CYAN}[%s%s]${NC} %3d%% (%s/%s images)" "$bar" "$bar_empty" "$pct" "$current" "$total"
                        fi
                    fi
                fi

                # Show spinner while model is loading (before any [i/total] progress appears)
                if [ "$progress_started" = false ]; then
                    spinner_char="${SPINNER_CHARS:$spinner_idx:1}"
                    # Show latest log status during loading phase
                    status_hint=""
                    if [ -f "$CAPTION_LOG" ]; then
                        recent_log=$(tail -n 10 "$CAPTION_LOG" 2>/dev/null)
                        # Show elapsed time alongside status
                        elapsed_min=$(( timeout_counter / 60 ))
                        elapsed_sec=$(( timeout_counter % 60 ))
                        if [ $elapsed_min -gt 0 ]; then
                            elapsed_str="${elapsed_min}m ${elapsed_sec}s"
                        else
                            elapsed_str="${elapsed_sec}s"
                        fi

                        if echo "$recent_log" | grep -q "Pre-loading model"; then
                            status_hint="Downloading and loading model... (${elapsed_str})"
                        elif echo "$recent_log" | grep -qE "Fetching [0-9]+ files"; then
                            fetch_info=$(echo "$recent_log" | grep -oE "Fetching [0-9]+ files" | tail -1)
                            status_hint="$fetch_info from HuggingFace... (${elapsed_str})"
                        elif echo "$recent_log" | grep -qE "Downloading model|downloading.*safetensors|\.safetensors"; then
                            status_hint="Downloading model weights... (${elapsed_str})"
                        elif echo "$recent_log" | grep -qE "HTTP Request|resolve"; then
                            status_hint="Downloading model from HuggingFace... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "Loading model"; then
                            status_hint="Loading caption model into GPU... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "Loading processor"; then
                            status_hint="Loading processor... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "Model loaded\|Model ready"; then
                            status_hint="Model loaded, starting captioning..."
                        elif echo "$recent_log" | grep -qE "Downloading.*\.whl"; then
                            pkg=$(echo "$recent_log" | grep -oE "Downloading [a-zA-Z_-]+" | tail -1 | sed 's/Downloading //')
                            if [ -n "$pkg" ]; then
                                status_hint="Downloading $pkg... (${elapsed_str})"
                            else
                                status_hint="Downloading dependencies... (${elapsed_str})"
                            fi
                        elif echo "$recent_log" | grep -q "Installing collected"; then
                            status_hint="Installing packages... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "Installing requirements"; then
                            status_hint="Installing requirements... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "Upgrading pip"; then
                            status_hint="Upgrading pip... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "Creating virtual environment"; then
                            status_hint="Creating virtual environment... (${elapsed_str})"
                        elif echo "$recent_log" | grep -qE "setup|Setting up"; then
                            status_hint="Setting up environment... (${elapsed_str})"
                        elif echo "$recent_log" | grep -q "CUDA"; then
                            status_hint="Checking CUDA compatibility..."
                        elif echo "$recent_log" | grep -qE "Collecting|Requirement already"; then
                            status_hint="Resolving dependencies... (${elapsed_str})"
                        elif echo "$recent_log" | grep -qE "Retrying|timed out"; then
                            status_hint="Network retry, downloading model... (${elapsed_str})"
                        else
                            # Show the last meaningful log line as context
                            last_msg=$(echo "$recent_log" | grep -oE "INFO - .*" | tail -1 | sed 's/INFO - //')
                            if [ -n "$last_msg" ] && [ ${#last_msg} -lt 60 ]; then
                                status_hint="$last_msg (${elapsed_str})"
                            else
                                status_hint="Setting up caption model... (${elapsed_str})"
                            fi
                        fi
                    else
                        status_hint="Starting..."
                    fi
                    printf "\r\033[K  ${CYAN}%s${NC} %s" "$spinner_char" "$status_hint"
                    spinner_idx=$(( (spinner_idx + 1) % ${#SPINNER_CHARS} ))
                fi

                sleep 2
                timeout_counter=$((timeout_counter + 2))
                if [ $timeout_counter -ge $max_timeout ]; then
                    echo ""
                    print_error "Image captioning timed out after 1 hour. Check log: $CAPTION_LOG"
                    exit 1
                fi
            done
            echo ""
            # Verify captioning actually completed successfully
            wait "$IMAGE_CAPTION_PID"
            if [ $? -ne 0 ]; then
                print_error "Image captioning failed. Check log: $NETWORK_VOLUME/logs/image_captioning.log"
                exit 1
            fi
            print_success "Image captioning completed!"
        else
            print_error "JoyCaption script not found at: $JOY_CAPTION_SCRIPT"
            print_info "This file should have been set up during pod initialization."
            print_info "Try restarting the pod to trigger the setup again."
            exit 1
        fi
    fi

    # Start video captioning if needed
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Starting video captioning process..."
        VIDEO_CAPTION_SCRIPT="$NETWORK_VOLUME/Captioning/video_captioner.sh"

        if [ -f "$VIDEO_CAPTION_SCRIPT" ]; then
            bash "$VIDEO_CAPTION_SCRIPT" > "$NETWORK_VOLUME/logs/video_captioning.log" 2>&1 &
            VIDEO_CAPTION_PID=$!

            # Wait for video captioning with progress indicator
            print_info "Waiting for video captioning to complete..."
            timeout_counter=0
            max_timeout=7200  # 2 hour timeout (videos take longer)
            while kill -0 "$VIDEO_CAPTION_PID" 2>/dev/null; do
                # Check for completion first
                if tail -n 1 "$NETWORK_VOLUME/logs/video_captioning.log" 2>/dev/null | grep -q "video captioning complete"; then
                    break
                fi
                # Check for actual errors (more specific patterns to avoid false positives)
                # Look for actual error patterns: [ERROR], Error:, Traceback, Exception:, or failed with exit code
                if tail -n 20 "$NETWORK_VOLUME/logs/video_captioning.log" 2>/dev/null | grep -qiE "(^\[ERROR\]|^Error:|^Traceback|Exception:|failed with exit)"; then
                    print_error "Video captioning encountered errors. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                    exit 1
                fi
                echo -n "."
                sleep 2
                timeout_counter=$((timeout_counter + 2))
                if [ $timeout_counter -ge $max_timeout ]; then
                    print_error "Video captioning timed out after 2 hours. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                    exit 1
                fi
            done
            echo ""

            wait "$VIDEO_CAPTION_PID"
            if [ $? -eq 0 ]; then
                print_success "Video captioning completed successfully"
            else
                print_error "Video captioning failed. Check log: $NETWORK_VOLUME/logs/video_captioning.log"
                exit 1
            fi
        else
            print_error "Video captioning script not found at: $VIDEO_CAPTION_SCRIPT"
            print_info "This file should have been set up during pod initialization."
            print_info "Try restarting the pod to trigger the setup again."
            exit 1
        fi
    fi

    echo ""
fi

# Wait for model download to complete
if [ -n "$MODEL_DOWNLOAD_PID" ]; then
    print_header "Finalizing Model Download"
    echo ""
    print_info "Downloading model..."
    echo ""

    DOWNLOAD_LOG="$NETWORK_VOLUME/logs/model_download.log"
    timeout_counter=0
    max_timeout=10800  # 3 hour timeout for large models
    BAR_WIDTH=30

    while kill -0 "$MODEL_DOWNLOAD_PID" 2>/dev/null; do
        # Check for auth/access errors in log (match actual error messages, not HTTP status codes in info lines)
        if tail -n 20 "$DOWNLOAD_LOG" 2>/dev/null | grep -qiE "(^error:|unauthorized|access denied|repository not found|401 Client Error|403 Client Error|404 Client Error)"; then
            echo ""
            print_error "Model download encountered errors."
            print_info "Check the full log: tail -n 50 $DOWNLOAD_LOG"
            print_info "Common causes:"
            print_info "  - Network connectivity issues (try restarting the pod)"
            print_info "  - Invalid or expired Hugging Face token"
            print_info "  - Insufficient disk space (check with: df -h)"
            kill "$MODEL_DOWNLOAD_PID" 2>/dev/null || true
            exit 1
        fi

        # Extract progress from hf download tqdm output
        # Matches patterns like: "Fetching 30 files:  47%|...| 14/30"
        # or individual file downloads: "model.safetensors:  65%|...| 1.30G/2.00G"
        if [ -f "$DOWNLOAD_LOG" ]; then
            progress_line=$(tail -c 4096 "$DOWNLOAD_LOG" 2>/dev/null | tr '\r' '\n' | grep -E '[0-9]+%\|' | tail -1)
            if [ -n "$progress_line" ]; then
                pct=$(echo "$progress_line" | grep -oE '[0-9]+%' | head -1 | tr -d '%')
                counts=$(echo "$progress_line" | grep -oE '[0-9]+/[0-9]+' | head -1)

                if [ -n "$pct" ]; then
                    filled=$(( pct * BAR_WIDTH / 100 ))
                    empty=$(( BAR_WIDTH - filled ))
                    bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null)
                    bar_empty=$(printf '%0.s░' $(seq 1 $empty 2>/dev/null) 2>/dev/null)

                    if [ -n "$counts" ]; then
                        printf "\r\033[K  ${CYAN}[%s%s]${NC} %3d%% (%s files)" "$bar" "$bar_empty" "$pct" "$counts"
                    else
                        printf "\r\033[K  ${CYAN}[%s%s]${NC} %3d%%" "$bar" "$bar_empty" "$pct"
                    fi
                fi
            fi
        fi

        sleep 2
        timeout_counter=$((timeout_counter + 2))
        if [ $timeout_counter -ge $max_timeout ]; then
            echo ""
            print_error "Model download timed out after 3 hours."
            print_info "This usually means the network is very slow or the download stalled."
            print_info "Check progress: tail -n 20 $DOWNLOAD_LOG"
            print_info "Try restarting the pod and running the script again — downloads resume from where they left off."
            kill "$MODEL_DOWNLOAD_PID" 2>/dev/null || true
            exit 1
        fi
    done
    echo ""
    wait "$MODEL_DOWNLOAD_PID"
    download_exit_code=$?

    if [ $download_exit_code -ne 0 ]; then
        print_error "Model download failed (exit code $download_exit_code)."
        print_info "Check the log for details: tail -n 50 $NETWORK_VOLUME/logs/model_download.log"
        print_info "You can restart the script to retry — downloads usually resume automatically."
        exit 1
    fi
    
    # Verify model files actually exist based on MODEL_TYPE
    print_info "Verifying model download..."
    model_verify_failed=false
    case $MODEL_TYPE in
        "flux")
            if [ ! -d "$NETWORK_VOLUME/models/flux" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/flux" 2>/dev/null)" ]; then
                print_error "Flux model files not found at: $NETWORK_VOLUME/models/flux"
                model_verify_failed=true
            fi
            ;;
        "sdxl")
            if [ ! -f "$NETWORK_VOLUME/models/sdXL_v10VAEFix.safetensors" ]; then
                print_error "SDXL model file not found at: $NETWORK_VOLUME/models/sdXL_v10VAEFix.safetensors"
                model_verify_failed=true
            fi
            ;;
        "wan13")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B" 2>/dev/null)" ]; then
                print_error "Wan 1.3B model files not found at: $NETWORK_VOLUME/models/Wan/Wan2.1-T2V-1.3B"
                model_verify_failed=true
            fi
            ;;
        "wan14b_t2v")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B" 2>/dev/null)" ]; then
                print_error "Wan 14B T2V model files not found at: $NETWORK_VOLUME/models/Wan/Wan2.1-T2V-14B"
                model_verify_failed=true
            fi
            ;;
        "wan14b_i2v")
            if [ ! -d "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P" 2>/dev/null)" ]; then
                print_error "Wan 14B I2V model files not found at: $NETWORK_VOLUME/models/Wan/Wan2.1-I2V-14B-480P"
                model_verify_failed=true
            fi
            ;;
        "qwen")
            if [ ! -d "$NETWORK_VOLUME/models/Qwen-Image" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Qwen-Image" 2>/dev/null)" ]; then
                print_error "Qwen Image model files not found at: $NETWORK_VOLUME/models/Qwen-Image"
                model_verify_failed=true
            fi
            ;;
        "qwen_2512")
            if [ ! -d "$NETWORK_VOLUME/models/Qwen-Image-2512" ] || [ -z "$(ls -A "$NETWORK_VOLUME/models/Qwen-Image-2512" 2>/dev/null)" ]; then
                print_error "Qwen Image 2512 model files not found at: $NETWORK_VOLUME/models/Qwen-Image-2512"
                model_verify_failed=true
            fi
            ;;
        "z_image_turbo")
            missing_files=""
            if [ ! -f "$NETWORK_VOLUME/models/z_image/z_image_turbo_bf16.safetensors" ]; then
                missing_files="$missing_files z_image_turbo_bf16.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/ae.safetensors" ]; then
                missing_files="$missing_files ae.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/qwen_3_4b.safetensors" ]; then
                missing_files="$missing_files qwen_3_4b.safetensors"
            fi
            if [ ! -f "$NETWORK_VOLUME/models/z_image/zimage_turbo_training_adapter_v2.safetensors" ]; then
                missing_files="$missing_files zimage_turbo_training_adapter_v2.safetensors"
            fi
            if [ -n "$missing_files" ]; then
                print_error "Z Image Turbo model files missing:$missing_files"
                model_verify_failed=true
            fi
            ;;
    esac

    if [ "$model_verify_failed" = true ]; then
        echo ""
        print_info "The download reported success but model files are missing or incomplete."
        print_info "Possible causes:"
        print_info "  - Insufficient disk space (check with: df -h)"
        print_info "  - Download was interrupted or corrupted"
        print_info "  - File permissions issue"
        print_info "Check the download log: tail -n 50 $NETWORK_VOLUME/logs/model_download.log"
        print_info "You can restart the script to retry — downloads usually resume automatically."
        exit 1
    fi
    print_success "Model download completed and verified!"
    echo ""
fi

# Update dataset.toml file with actual paths and video config
print_header "Configuring Dataset"
echo ""

DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"

if [ -f "$DATASET_TOML" ]; then
    print_info "Updating dataset.toml with actual paths..."

    # Create backup
    cp "$DATASET_TOML" "$DATASET_TOML.backup"

    # Replace $NETWORK_VOLUME with actual path in image directory
    sed -i "s|\$NETWORK_VOLUME/image_dataset_here|$NETWORK_VOLUME/image_dataset_here|g" "$DATASET_TOML" 2>/dev/null || print_warning "Failed to update image directory path in dataset.toml"

    # Replace $NETWORK_VOLUME with actual path in video directory (even if commented)
    sed -i "s|\$NETWORK_VOLUME/video_dataset_here|$NETWORK_VOLUME/video_dataset_here|g" "$DATASET_TOML" 2>/dev/null || print_warning "Failed to update video directory path in dataset.toml"

    # Uncomment video dataset section if user wants to caption videos
    if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
        print_info "Enabling video dataset in configuration..."
        # Uncomment the video directory section
        sed -i '/# \[\[directory\]\]/,/# num_repeats = 5/ s/^# //' "$DATASET_TOML" 2>/dev/null
        # Verify uncommenting worked by checking if video directory section exists uncommented
        if ! grep -q "^\[\[directory\]\]" "$DATASET_TOML" || [ -z "$(grep -A2 "^\[\[directory\]\]" "$DATASET_TOML" | grep -m1 "video_dataset_here")" ]; then
            # Check if there's a commented video section that wasn't uncommented
            if grep -q "# path = '\$NETWORK_VOLUME/video_dataset_here'" "$DATASET_TOML"; then
                print_warning "Video dataset section may not have been uncommented correctly. Please check dataset.toml manually."
            fi
        fi
    fi

    print_success "Dataset configuration updated"
else
    print_warning "dataset.toml not found at $DATASET_TOML"
fi

# Also configure dataset_qwen.toml for Qwen Image 2512 (uses 1328 resolution)
if [ "$MODEL_TYPE" = "qwen_2512" ]; then
    DATASET_QWEN_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset_qwen.toml"
    if [ -f "$DATASET_QWEN_TOML" ]; then
        print_info "Updating dataset_qwen.toml with actual paths..."
        cp "$DATASET_QWEN_TOML" "$DATASET_QWEN_TOML.backup"
        sed -i "s|\$NETWORK_VOLUME/image_dataset_here|$NETWORK_VOLUME/image_dataset_here|g" "$DATASET_QWEN_TOML" 2>/dev/null || print_warning "Failed to update image directory path in dataset_qwen.toml"
        sed -i "s|\$NETWORK_VOLUME/video_dataset_here|$NETWORK_VOLUME/video_dataset_here|g" "$DATASET_QWEN_TOML" 2>/dev/null || print_warning "Failed to update video directory path in dataset_qwen.toml"
        if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
            print_info "Enabling video dataset in Qwen dataset configuration..."
            sed -i '/# \[\[directory\]\]/,/# num_repeats = 5/ s/^# //' "$DATASET_QWEN_TOML" 2>/dev/null
        fi
        print_success "Qwen dataset configuration updated (resolution: 1328)"
    else
        print_warning "dataset_qwen.toml not found at $DATASET_QWEN_TOML"
    fi
fi

# Extract and display training configuration summary
print_header "Training Configuration Summary"
echo ""

# Read resolution from dataset.toml (use dataset_qwen.toml for Qwen 2512)
if [ "$MODEL_TYPE" = "qwen_2512" ] && [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/dataset_qwen.toml" ]; then
    RESOLUTION=$(grep "^resolutions = " "$NETWORK_VOLUME/diffusion_pipe/examples/dataset_qwen.toml" | sed 's/resolutions = \[\([0-9]*\)\]/\1/')
    if [ -z "$RESOLUTION" ]; then
        RESOLUTION="1328"
    fi
elif [ -f "$DATASET_TOML" ]; then
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

echo -e "${BOLD}Model:${NC} $MODEL_NAME"
echo -e "${BOLD}TOML Config:${NC} examples/$TOML_FILE"
# Only show resolution as WxH if it's a number, otherwise show as-is
if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
    echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}x${RESOLUTION}"
else
    echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}"
fi
echo ""

echo -e "${BOLD}Training Parameters:${NC}"
echo "  📊 Epochs: $EPOCHS"
echo "  💾 Save Every: $SAVE_EVERY epochs"
echo "  🎛️  LoRA Rank: $RANK"
echo "  📈 Learning Rate: $LR"
echo "  ⚙️  Optimizer: $OPTIMIZER_TYPE"
echo ""

# Show dataset paths and repeats
if [ "$CAPTION_MODE" != "skip" ]; then
    echo -e "${BOLD}Dataset Configuration:${NC}"

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
    echo -e "${BOLD}Dataset:${NC} Using existing captions"
fi

if [ "$MODEL_TYPE" = "flux" ]; then
    echo -e "${BOLD}Hugging Face Token:${NC} Set ✓"
fi

if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    echo -e "${BOLD}Gemini API Key:${NC} Set ✓"
fi

echo ""

# Prompt user about configuration files
print_header "Training Configuration"
echo ""

print_info "Before starting training, you can modify the default training parameters in these files:"
echo ""
echo -e "${BOLD}1. Model Configuration:${NC}"
echo "   $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
echo ""
echo -e "${BOLD}2. Dataset Configuration:${NC}"
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
            print_info "Training paused for manual configuration."
            echo ""
            echo -e "${BOLD}Configuration Files:${NC}"
            echo "1. Model settings: $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
            echo "2. Dataset settings: $NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
            echo ""
            print_warning "Please modify these files as needed, then return here to continue."
            echo ""

            while true; do
                read -p "Have you finished configuring the settings? (yes/no): " config_done
                case $config_done in
                    yes|YES|y|Y)
                        print_success "Configuration completed. Reading updated settings..."
                        echo ""

                        # Re-read training parameters from updated TOML files
                        MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
                        DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"

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

                        # Display updated configuration for confirmation
                        print_header "Updated Training Configuration"
                        echo ""
                        echo -e "${BOLD}Model:${NC} $MODEL_NAME"
                        # Only show resolution as WxH if it's a number, otherwise show as-is
                        if [[ "$RESOLUTION" =~ ^[0-9]+$ ]]; then
                            echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}x${RESOLUTION}"
                        else
                            echo -e "${BOLD}Resolution:${NC} ${RESOLUTION}"
                        fi
                        echo ""
                        echo -e "${BOLD}Updated Training Parameters:${NC}"
                        echo "  📊 Epochs: $EPOCHS"
                        echo "  💾 Save Every: $SAVE_EVERY epochs"
                        echo "  🎛️  LoRA Rank: $RANK"
                        echo "  📈 Learning Rate: $LR"
                        echo "  ⚙️  Optimizer: $OPTIMIZER_TYPE"
                        echo ""

                        while true; do
                            read -p "Do these updated settings look correct? (yes/no): " settings_confirm
                            case $settings_confirm in
                                yes|YES|y|Y)
                                    print_success "Settings confirmed. Proceeding with training..."
                                    break 2  # Break out of both loops
                                    ;;
                                no|NO|n|N)
                                    print_info "Please modify the configuration files again."
                                    echo ""
                                    break  # Go back to configuration loop
                                    ;;
                                *)
                                    print_error "Please enter 'yes' or 'no'."
                                    ;;
                            esac
                        done
                        ;;
                    no|NO|n|N)
                        print_info "Take your time configuring the settings."
                        ;;
                    *)
                        print_error "Please enter 'yes' or 'no'."
                        ;;
                esac
            done
            break
            ;;
        *)
            print_error "Invalid choice. Please enter 1 or 2."
            ;;
    esac
done

echo ""

# Check if image captioning is still running
if [ "$CAPTION_MODE" = "images" ] || [ "$CAPTION_MODE" = "both" ]; then
    # Image captioning was already handled in the captioning section above
    # No need to check again here

    # Prompt user to inspect image captions
    print_header "Caption Inspection"
    echo ""
    print_info "Please manually inspect the generated captions in:"
    echo "  $NETWORK_VOLUME/image_dataset_here"
    echo ""
    print_warning "Check that the captions are accurate and appropriate for your training data."
    echo ""

    while true; do
        read -p "Have you reviewed the image captions and are ready to proceed? (yes/no): " inspect_choice
        case $inspect_choice in
            yes|YES|y|Y)
                print_success "Image captions approved. Proceeding to training..."
                break
                ;;
            no|NO|n|N)
                print_info "Please review the captions and run this script again when ready."
                exit 0
                ;;
            *)
                print_error "Please enter 'yes' or 'no'."
                ;;
        esac
    done
    echo ""
fi

# Check video captions if applicable
if [ "$CAPTION_MODE" = "videos" ] || [ "$CAPTION_MODE" = "both" ]; then
    # Video captioning was already handled in the captioning section above
    # No need to check again here

    print_header "Video Caption Inspection"
    echo ""
    print_info "Please manually inspect the generated video captions in:"
    echo "  $NETWORK_VOLUME/video_dataset_here"
    echo ""
    print_warning "Check that the video captions are accurate and appropriate for your training data."
    echo ""

    while true; do
        read -p "Have you reviewed the video captions and are ready to proceed? (yes/no): " video_inspect_choice
        case $video_inspect_choice in
            yes|YES|y|Y)
                print_success "Video captions approved. Proceeding to training..."
                break
                ;;
            no|NO|n|N)
                print_info "Please review the captions and run this script again when ready."
                exit 0
                ;;
            *)
                print_error "Please enter 'yes' or 'no'."
                ;;
        esac
    done
    echo ""
fi

# Start training
print_header "Starting Training"
echo ""

print_info "Changing to diffusion_pipe directory..."
if ! cd "$NETWORK_VOLUME/diffusion_pipe"; then
    print_error "Could not find the diffusion_pipe directory at: $NETWORK_VOLUME/diffusion_pipe"
    print_error "This usually means the workspace was not set up correctly."
    print_error "Try restarting the pod to trigger the initial setup again."
    exit 1
fi

# Verify the training config file exists before proceeding
if [ ! -f "examples/$TOML_FILE" ]; then
    print_error "Training configuration file not found: examples/$TOML_FILE"
    print_error "Expected location: $NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
    print_error "This file should have been copied during model setup. Try selecting the model again."
    exit 1
fi

# Dependencies are pinned at pod boot (src/start.sh) as a single internally-consistent set.
# Do NOT re-float transformers/peft here — `pip install transformers -U` pulls transformers 5.x,
# which removed CLIPTextModel.text_model and breaks SDXL/CLIP loading (verified Phase-0 2026-06-16).
print_info "Using the pinned dependency set installed at boot (transformers 4.57.6, peft 0.19.1)."

echo ""

print_info "Starting LoRA training with $MODEL_NAME..."
print_info "Using configuration: examples/$TOML_FILE"
echo ""

# Add special warning for Qwen Image model initialization
if [ "$MODEL_TYPE" = "qwen" ] || [ "$MODEL_TYPE" = "qwen_2512" ]; then
    print_warning "⚠️  IMPORTANT: Qwen Image model initialization can take several minutes."
    print_warning "⚠️  The script may appear to hang during initialization - this is NORMAL."
    print_warning "⚠️  As long as the script doesn't exit with an error, let it run."
    echo ""
    print_info "Waiting 10 seconds for you to read this message..."
    sleep 10
    echo ""
fi

# Add special warning for Z Image Turbo model initialization
if [ "$MODEL_TYPE" = "z_image_turbo" ]; then
    print_warning "⚠️  IMPORTANT: Z Image Turbo model initialization can take several minutes."
    print_warning "⚠️  The script may appear to hang during initialization - this is NORMAL."
    print_warning "⚠️  As long as the script doesn't exit with an error, let it run."
    echo ""
    print_info "Waiting 10 seconds for you to read this message..."
    sleep 10
    echo ""
fi

print_warning "Training is starting. This may take several hours depending on your dataset size and model."
print_info "You can monitor progress in the console output below."
echo ""

# Refuse to train if boot flagged the environment as degraded. start.sh diagnoses pinned-dependency
# problems (wrong torch, missing deepspeed, comfy-kitchen stub) and appends them to this sentinel;
# this is the real gate, at the point training is actually launched.
if [ -f /tmp/ENV_DEGRADED ]; then
    print_error "REFUSING TO TRAIN — environment is degraded:"
    cat /tmp/ENV_DEGRADED
    echo "Fix the above (or rebuild the pod) before training. Aborting."
    exit 1
fi

# Start training with the appropriate TOML file
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config "examples/$TOML_FILE"

print_success "Training completed!"