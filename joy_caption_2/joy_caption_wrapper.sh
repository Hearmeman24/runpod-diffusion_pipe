#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/joy_caption_env"
PYTHON_SCRIPT="$SCRIPT_DIR/joy_caption_batch.py"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

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
    echo "Usage: $0 INPUT_DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  INPUT_DIR                    Directory containing images to process"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR             Directory to save caption files (default: same as input)"
    echo "  --prompt TEXT                Caption generation prompt (default: 'Write a descriptive caption for this image.')"
    echo "  --trigger-word WORD          Trigger word to prepend to captions (e.g., 'claude' -> 'claude, <caption>')"
    echo "  --no-skip-existing           Process all images even if caption files already exist"
    echo "  --timeout MINUTES            Model unload timeout in minutes (default: 5)"
    echo "  --setup-only                 Only setup the environment, don't run captioning"
    echo "  --force-reinstall            Force reinstall of all requirements"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/images"
    echo "  $0 /path/to/images --trigger-word 'claude' --output-dir /path/to/captions"
    echo "  $0 /path/to/images --prompt 'Describe this image in detail.' --timeout 10"
    echo "  $0 --setup-only              # Just setup the environment"
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

    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment at $VENV_DIR"
        $PYTHON_CMD -m venv "$VENV_DIR"
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
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
        *)
            SCRIPT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting Joy Caption Batch Processing Setup"

    # Check Python installation
    check_python

    # Create requirements file if needed
    create_requirements

    # Setup virtual environment
    setup_venv

    # If setup-only flag is set, exit here
    if [[ "$SETUP_ONLY" == "true" ]]; then
        log_success "Environment setup complete. You can now run the script normally."
        exit 0
    fi

    # Check if we have arguments to process
    if [[ ${#SCRIPT_ARGS[@]} -eq 0 ]]; then
        log_error "No input directory specified"
        echo ""
        show_help
        exit 1
    fi

    # Check if Python script exists
    check_python_script

    # Run the captioning script with all arguments
    run_captioning "${SCRIPT_ARGS[@]}"

    log_success "All done! 🎉"
}

# Run main function
main "$@"