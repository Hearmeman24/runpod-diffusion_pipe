#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/joy_caption_env"
PYTHON_SCRIPT="$SCRIPT_DIR/joy_caption_batch.py"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# Default image directory detection
detect_default_image_dir() {
    # First check if NETWORK_VOLUME is set
    if [[ -n "$NETWORK_VOLUME" && -d "$NETWORK_VOLUME/image_dataset_here" ]]; then
        echo "$NETWORK_VOLUME/image_dataset_here"
    # Check for workspace volume
    elif [[ -d "/workspace/diffusion_pipe_working_folder/image_dataset_here" ]]; then
        echo "/workspace/diffusion_pipe_working_folder/image_dataset_here"
    # Check for local volume
    elif [[ -d "/diffusion_pipe_working_folder/image_dataset_here" ]]; then
        echo "/diffusion_pipe_working_folder/image_dataset_here"
    # Fallback to current directory
    else
        echo "."
    fi
}

DEFAULT_IMAGE_DIR=$(detect_default_image_dir)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_help() {
    echo "Joy Caption Batch Processing Wrapper"
    echo ""
    echo "This script sets up a virtual environment and runs the Joy Caption batch processor."
    echo ""
    echo "Usage: $0 [INPUT_DIR] [OPTIONS]"
    echo ""
    echo "Optional:"
    echo "  INPUT_DIR                    Directory containing images to process"
    echo "                              (default: $DEFAULT_IMAGE_DIR)"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR             Directory to save caption files (default: same as input)"
    echo "  --prompt TEXT                Caption generation prompt (default: Write a descriptive caption for this image in a casual tone within 50 words. Do NOT mention any text that is in the image."
    echo "  --trigger-word WORD          Trigger word to prepend to captions (e.g., 'Alice' -> 'Alice, <caption>')"
    echo "  --no-skip-existing           Process all images even if caption files already exist"
    echo "  --setup-only                 Only setup the environment, don't run captioning"
    echo "  --force-reinstall            Force reinstall of all requirements"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Use default image directory: $DEFAULT_IMAGE_DIR"
    echo "  $0 /path/to/images"
    echo "  $0 --trigger-word 'claude' --output-dir /path/to/captions"
    echo "  $0 /path/to/images --prompt 'Describe this image in detail.' --timeout 10"
    echo "  $0 --setup-only              # Just setup the environment"
    echo ""
    echo "Default image directory detection:"
    echo "  1. \$NETWORK_VOLUME/image_dataset_here (if NETWORK_VOLUME is set)"
    echo "  2. /workspace/diffusion_pipe_working_folder/image_dataset_here"
    echo "  3. /diffusion_pipe_working_folder/image_dataset_here"
    echo "  4. Current directory (.)"
}

# Function to install system dependencies
install_system_deps() {
    log_info "Checking system dependencies..."

    # Check if we need to install python3-venv
    if ! dpkg -l | grep -q python3-venv 2>/dev/null; then
        log_info "Installing required system packages..."

        # Update package list
        if command -v apt &> /dev/null; then
            log_info "Updating package list..."
            apt update || {
                log_warning "Failed to update package list. Continuing anyway..."
            }

            log_info "Installing python3-venv..."
            apt install -y python3-venv || {
                log_error "Failed to install python3-venv. You may need to run this script as root or with sudo."
                exit 1
            }
            log_success "System dependencies installed"
        else
            log_warning "apt package manager not found. Assuming dependencies are available."
        fi
    else
        log_info "System dependencies already installed"
    fi
}

# Function to check if Python 3.8+ is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.8"

    if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        log_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi

    log_info "Using Python $PYTHON_VERSION: $($PYTHON_CMD --version)"
}

# Function to create requirements.txt if it doesn't exist
create_requirements() {
    if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
        log_info "Creating requirements.txt file..."
        cat > "$REQUIREMENTS_FILE" << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
accelerate>=0.20.0
Pillow>=9.0.0
numpy>=1.21.0
safetensors>=0.3.0
sentencepiece>=0.1.99
protobuf>=3.20.0
EOF
        log_success "Created requirements.txt"
    fi
}

# Function to setup virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."

    # Remove existing venv if force reinstall is requested
    if [[ "$FORCE_REINSTALL" == "true" && -d "$VENV_DIR" ]]; then
        log_warning "Removing existing virtual environment for fresh install..."
        rm -rf "$VENV_DIR"
    fi

    # Check if venv exists and is valid
    if [[ -d "$VENV_DIR" ]]; then
        if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
            log_warning "Virtual environment appears corrupted (missing activate script)"
            log_info "Removing corrupted virtual environment..."
            rm -rf "$VENV_DIR"
        else
            log_info "Virtual environment already exists and appears valid"
        fi
    fi

    # Create virtual environment if it doesn't exist or was removed
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment at $VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"

        # Verify creation was successful
        if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
            log_error "Failed to create virtual environment properly"
            log_error "The activate script was not created"
            exit 1
        fi

        log_success "Virtual environment created successfully"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

    # Install or upgrade requirements
    if [[ "$FORCE_REINSTALL" == "true" ]]; then
        log_info "Force reinstalling requirements..."
        pip install --force-reinstall -r "$REQUIREMENTS_FILE"
    else
        log_info "Installing requirements..."
        pip install -r "$REQUIREMENTS_FILE"
    fi

    log_success "Virtual environment setup complete"
}

# Function to check if the Python script exists
check_python_script() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        log_error "Please ensure joy_caption_batch.py is in the same directory as this script"
        exit 1
    fi
}

# Function to run the captioning script
run_captioning() {
    log_info "Activating virtual environment and running Joy Caption batch processor..."

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Build command with all passed arguments
    CMD="python \"$PYTHON_SCRIPT\""

    # Add all arguments passed to this script
    for arg in "$@"; do
        CMD="$CMD \"$arg\""
    done

    log_info "Running command: $CMD"

    # Execute the command
    eval $CMD

    log_success "Joy Caption processing completed"
}

# Parse command line arguments
SETUP_ONLY=false
FORCE_REINSTALL=false
SCRIPT_ARGS=()
INPUT_DIR_SPECIFIED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --force-reinstall)
            FORCE_REINSTALL=true
            shift
            ;;
        --output-dir|--prompt|--trigger-word|--timeout)
            # These are options that take values, add both the option and its value
            SCRIPT_ARGS+=("$1")
            shift
            if [[ $# -gt 0 ]]; then
                SCRIPT_ARGS+=("$1")
                shift
            fi
            ;;
        --no-skip-existing)
            # This is a flag option
            SCRIPT_ARGS+=("$1")
            shift
            ;;
        *)
            # Check if this looks like a directory path (first positional argument)
            if [[ ! "$INPUT_DIR_SPECIFIED" == "true" && ! "$1" =~ ^-- ]]; then
                INPUT_DIR_SPECIFIED=true
                SCRIPT_ARGS+=("$1")
            else
                SCRIPT_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting Joy Caption Batch Processing Setup"

    # Install system dependencies first
    install_system_deps

    # Check Python installation
    check_python

    # CUDA compatibility check
    check_cuda_compatibility() {
        $PYTHON_CMD << 'PYTHON_EOF'
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

    # Create requirements file if needed
    create_requirements

    # Setup virtual environment
    setup_venv

    # If setup-only flag is set, exit here
    if [[ "$SETUP_ONLY" == "true" ]]; then
        log_success "Environment setup complete. You can now run the script normally."
        exit 0
    fi

    # Check if Python script exists
    check_python_script

    # If no input directory was specified, use the default
    if [[ "$INPUT_DIR_SPECIFIED" == "false" ]]; then
        log_info "No input directory specified, using default: $DEFAULT_IMAGE_DIR"

        # Check if the default directory exists
        if [[ ! -d "$DEFAULT_IMAGE_DIR" ]]; then
            log_warning "Default image directory does not exist: $DEFAULT_IMAGE_DIR"
            log_info "Creating directory: $DEFAULT_IMAGE_DIR"
            mkdir -p "$DEFAULT_IMAGE_DIR" || {
                log_error "Failed to create directory: $DEFAULT_IMAGE_DIR"
                exit 1
            }
        fi

        # Add the default directory as the first argument
        SCRIPT_ARGS=("$DEFAULT_IMAGE_DIR" "${SCRIPT_ARGS[@]}")
    fi

    # Run the captioning script with all arguments
    run_captioning "${SCRIPT_ARGS[@]}"

    log_success "All done! 🎉"
}

# Run main function
main "$@"