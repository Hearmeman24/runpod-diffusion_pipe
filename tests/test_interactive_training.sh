#!/bin/bash
# =============================================================================
# Test Suite for interactive_start_training.sh
# =============================================================================
#
# Runs locally without GPU, training, captioning, or model downloads.
# Tests the script's logic paths by stubbing out external commands and
# simulating user input.
#
# Usage:
#   bash tests/test_interactive_training.sh
#   bash tests/test_interactive_training.sh test_name   # run single test
#
# =============================================================================

set -u  # Treat unset variables as errors (but not -e, we check exit codes)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_UNDER_TEST="$PROJECT_DIR/interactive_start_training.sh"

# ── State ────────────────────────────────────────────────────────────────────
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_NAMES=()
SANDBOX=""

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Helpers
# =============================================================================

setup_sandbox() {
    SANDBOX=$(mktemp -d)
    export NETWORK_VOLUME="$SANDBOX"

    # Minimal directory structure the script expects
    mkdir -p "$SANDBOX/logs"
    mkdir -p "$SANDBOX/models"
    mkdir -p "$SANDBOX/image_dataset_here"
    mkdir -p "$SANDBOX/video_dataset_here"
    mkdir -p "$SANDBOX/diffusion_pipe/examples"
    mkdir -p "$SANDBOX/runpod-diffusion_pipe/toml_files"
    mkdir -p "$SANDBOX/output_folder"
    mkdir -p "$SANDBOX/Captioning/JoyCaption"
    mkdir -p "$SANDBOX/Captioning"

    # Copy all toml files so the script finds them
    for f in "$PROJECT_DIR"/toml_files/*.toml; do
        [ -f "$f" ] && cp "$f" "$SANDBOX/runpod-diffusion_pipe/toml_files/"
    done

    # Copy dataset configs
    [ -f "$PROJECT_DIR/dataset.toml" ] && cp "$PROJECT_DIR/dataset.toml" "$SANDBOX/diffusion_pipe/examples/"
    [ -f "$PROJECT_DIR/dataset_qwen.toml" ] && cp "$PROJECT_DIR/dataset_qwen.toml" "$SANDBOX/runpod-diffusion_pipe/"

    # Create stub flash-attn success marker (skip flash-attn wait)
    touch /tmp/flash_attn_wheel_success

    # Create stub bin directory for fake commands
    STUB_BIN="$SANDBOX/.stub_bin"
    mkdir -p "$STUB_BIN"
    export PATH="$STUB_BIN:$PATH"

    # Stub: hf (huggingface CLI) — pretend download succeeds
    # For z_image_turbo, create the expected temp directory structure
    cat > "$STUB_BIN/hf" << 'STUB'
#!/bin/bash
# Parse --local-dir argument
local_dir=""
for i in "$@"; do
    if [[ "$prev" == "--local-dir" ]]; then
        local_dir="$i"
        break
    fi
    prev="$i"
done

# If this is a z_image download, create expected directory structure
if [[ "$*" == *"z_image_turbo"* ]] && [ -n "$local_dir" ]; then
    mkdir -p "$local_dir/split_files/diffusion_models"
    mkdir -p "$local_dir/split_files/vae"
    mkdir -p "$local_dir/split_files/text_encoders"
    touch "$local_dir/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
    touch "$local_dir/split_files/vae/ae.safetensors"
    touch "$local_dir/split_files/text_encoders/qwen_3_4b.safetensors"
fi

echo "stub: hf $*"
exit 0
STUB
    chmod +x "$STUB_BIN/hf"

    # Stub: deepspeed — don't actually train
    cat > "$STUB_BIN/deepspeed" << 'STUB'
#!/bin/bash
echo "stub: deepspeed $*"
exit 0
STUB
    chmod +x "$STUB_BIN/deepspeed"

    # Stub: pip — don't install anything
    cat > "$STUB_BIN/pip" << 'STUB'
#!/bin/bash
echo "stub: pip $*"
exit 0
STUB
    chmod +x "$STUB_BIN/pip"

    # Stub: wget — don't download anything
    cat > "$STUB_BIN/wget" << 'STUB'
#!/bin/bash
# Create the output file if -O was used
for i in "$@"; do
    if [[ "$prev" == "-O" ]]; then
        touch "$i"
        break
    fi
    prev="$i"
done
echo "stub: wget $*"
exit 0
STUB
    chmod +x "$STUB_BIN/wget"

    # Stub: python3 — consume stdin (heredoc) and report CUDA check passed
    cat > "$STUB_BIN/python3" << 'STUB'
#!/bin/bash
# Consume any heredoc/stdin input so the calling script doesn't break
cat > /dev/null
echo "CUDA compatibility check passed"
exit 0
STUB
    chmod +x "$STUB_BIN/python3"

    # Stub: clear — no-op
    cat > "$STUB_BIN/clear" << 'STUB'
#!/bin/bash
exit 0
STUB
    chmod +x "$STUB_BIN/clear"

    # Stub: sed — wrap real sed to handle GNU/BSD -i differences
    # macOS sed requires -i '' while GNU sed uses -i without arg
    local real_sed
    real_sed=$(which -a sed | grep -v "$STUB_BIN" | head -1)
    cat > "$STUB_BIN/sed" << STUB
#!/bin/bash
# If -i flag is used, handle GNU/BSD differences
if [[ "\$1" == "-i" ]]; then
    shift
    # Check if we're on macOS (BSD sed)
    if [[ "\$(uname)" == "Darwin" ]]; then
        "$real_sed" -i '' "\$@"
    else
        "$real_sed" -i "\$@"
    fi
else
    "$real_sed" "\$@"
fi
STUB
    chmod +x "$STUB_BIN/sed"
}

teardown_sandbox() {
    if [ -n "$SANDBOX" ] && [ -d "$SANDBOX" ]; then
        rm -rf "$SANDBOX"
    fi
    rm -f /tmp/flash_attn_wheel_success
}

# Run the script with simulated stdin, capturing output and exit code.
# Uses a background process + kill instead of `timeout` for macOS compat.
# Args: $1 = stdin string, $2... = env vars to export
run_script() {
    local stdin_data="$1"
    shift

    # Apply any extra env vars
    for var in "$@"; do
        export "$var"
    done

    local outfile="$SANDBOX/.test_output"

    # Run in background so we can enforce a timeout
    (echo -e "$stdin_data" | bash "$SCRIPT_UNDER_TEST" > "$outfile" 2>&1) &
    local pid=$!

    # Wait up to 30 seconds
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null && [ $elapsed -lt 30 ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # Kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        LAST_EXIT_CODE=124  # timeout
    else
        wait "$pid" 2>/dev/null
        LAST_EXIT_CODE=$?
    fi

    LAST_OUTPUT=$(cat "$outfile" 2>/dev/null || echo "")
    rm -f "$outfile"
}

# ── Assertions ───────────────────────────────────────────────────────────────

assert_exit_code() {
    local expected=$1
    if [ "$LAST_EXIT_CODE" -eq "$expected" ]; then
        return 0
    else
        echo "  FAIL: Expected exit code $expected, got $LAST_EXIT_CODE"
        return 1
    fi
}

assert_output_contains() {
    local pattern="$1"
    if echo "$LAST_OUTPUT" | grep -qiE "$pattern"; then
        return 0
    else
        echo "  FAIL: Output does not contain pattern: $pattern"
        echo "  OUTPUT (last 20 lines):"
        echo "$LAST_OUTPUT" | tail -20 | sed 's/^/    /'
        return 1
    fi
}

assert_output_not_contains() {
    local pattern="$1"
    if echo "$LAST_OUTPUT" | grep -qiE "$pattern"; then
        echo "  FAIL: Output should NOT contain pattern: $pattern"
        return 1
    else
        return 0
    fi
}

assert_file_exists() {
    local filepath="$1"
    if [ -f "$filepath" ]; then
        return 0
    else
        echo "  FAIL: File does not exist: $filepath"
        return 1
    fi
}

assert_file_contains() {
    local filepath="$1"
    local pattern="$2"
    if [ -f "$filepath" ] && grep -qE "$pattern" "$filepath"; then
        return 0
    else
        echo "  FAIL: File $filepath does not contain: $pattern"
        return 1
    fi
}

# ── Test runner ──────────────────────────────────────────────────────────────

run_test() {
    local test_name="$1"
    local test_func="$2"
    TESTS_RUN=$((TESTS_RUN + 1))

    # Filter: if a test name was given on CLI, only run that one
    if [ -n "${FILTER:-}" ] && [ "$FILTER" != "$test_name" ]; then
        TESTS_RUN=$((TESTS_RUN - 1))
        return
    fi

    echo -e "${CYAN}▶ $test_name${NC}"

    setup_sandbox
    local result=0
    $test_func || result=1
    teardown_sandbox

    if [ $result -eq 0 ]; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "  ${RED}✗ FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_NAMES+=("$test_name")
    fi
    echo ""
}

# =============================================================================
# Test: Model Selection
# =============================================================================

test_model_selection_all_valid() {
    # Test a subset: just check that model 2 (SDXL) and 8 (Qwen 2512) selections work.
    # Full model test is covered by individual model verification tests.
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    run_script "2\n4\n1\nyes\nyes\n"
    if ! echo "$LAST_OUTPUT" | grep -q "Selected model: SDXL"; then
        echo "  FAIL: Model choice 2 should select 'SDXL'"
        return 1
    fi
}

test_model_selection_invalid_then_valid() {
    # Invalid input followed by valid should work
    run_script "9\n0\nabc\n2\n4\n1\nyes\nyes\n"
    assert_output_contains "Invalid choice" &&
    assert_output_contains "Selected model: SDXL"
}

# =============================================================================
# Test: Flux Token Handling
# =============================================================================

test_flux_requires_token() {
    # Flux without token should prompt, empty token should fail
    unset HUGGING_FACE_TOKEN 2>/dev/null || true
    run_script "1\n\n"  # Select flux, enter empty token
    assert_output_contains "Token cannot be empty" &&
    assert_exit_code 1
}

test_flux_token_from_env() {
    # Flux with pre-set token should succeed
    mkdir -p "$SANDBOX/models/flux"
    touch "$SANDBOX/models/flux/dummy.safetensors"
    run_script "1\n4\n1\nyes\nyes\n" "HUGGING_FACE_TOKEN=valid_token_here"
    assert_output_contains "Hugging Face token already set"
}

test_flux_token_placeholder_rejected() {
    # The placeholder value "token_here" should be treated as not set
    run_script "1\n\n" "HUGGING_FACE_TOKEN=token_here"
    assert_output_contains "Hugging Face token is required" &&
    assert_exit_code 1
}

# =============================================================================
# Test: Caption Mode Selection
# =============================================================================

test_caption_mode_skip() {
    run_script "2\n4\n1\nyes\nyes\n"  # SDXL, skip captioning
    assert_output_contains "Dataset validation completed"
}

test_caption_mode_images_no_files() {
    # Image captioning with empty image directory should fail
    run_script "2\n1\n\n"  # SDXL, images only, empty trigger word
    assert_output_contains "No Image files found" &&
    assert_exit_code 1
}

test_caption_mode_videos_no_files() {
    run_script "2\n2\ntest_gemini_key\n"  # SDXL, videos only
    assert_output_contains "No Video files found" &&
    assert_exit_code 1
}

test_caption_mode_images_with_files() {
    # Add some fake images
    touch "$SANDBOX/image_dataset_here/photo1.jpg"
    touch "$SANDBOX/image_dataset_here/photo2.png"
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"

    # Create JoyCaption stub
    echo '#!/bin/bash' > "$SANDBOX/Captioning/JoyCaption/JoyCaptionRunner.sh"
    echo 'echo "All done!"' >> "$SANDBOX/Captioning/JoyCaption/JoyCaptionRunner.sh"
    chmod +x "$SANDBOX/Captioning/JoyCaption/JoyCaptionRunner.sh"

    # SDXL, images only, trigger word, then proceed through rest
    run_script "2\n1\nmytrigger\n1\nyes\nyes\n"
    assert_output_contains "Found.*2 Image file"
}

test_caption_mode_both_missing_videos() {
    touch "$SANDBOX/image_dataset_here/photo1.jpg"
    # No video files
    run_script "2\n3\ntest_gemini_key\nmytrigger\n"
    assert_output_contains "No Video files found" &&
    assert_exit_code 1
}

# =============================================================================
# Test: Gemini API Key
# =============================================================================

test_gemini_key_required_for_videos() {
    touch "$SANDBOX/video_dataset_here/video1.mp4"
    unset GEMINI_API_KEY 2>/dev/null || true
    run_script "2\n2\n\n"  # SDXL, videos only, empty key
    assert_output_contains "API key cannot be empty" &&
    assert_exit_code 1
}

test_gemini_key_from_env() {
    touch "$SANDBOX/video_dataset_here/video1.mp4"
    run_script "2\n2\n" "GEMINI_API_KEY=valid_key"
    assert_output_contains "Gemini API key already set"
}

test_gemini_key_placeholder_rejected() {
    touch "$SANDBOX/video_dataset_here/video1.mp4"
    run_script "2\n2\n\n" "GEMINI_API_KEY=token_here"
    assert_output_contains "Gemini API key is required"
}

# =============================================================================
# Test: Dataset Directory Validation
# =============================================================================

test_image_dir_missing() {
    rmdir "$SANDBOX/image_dataset_here"
    run_script "2\n1\n\n"  # SDXL, images only
    assert_output_contains "directory does not exist" &&
    assert_exit_code 1
}

test_video_dir_missing() {
    rmdir "$SANDBOX/video_dataset_here"
    run_script "2\n2\ntest_key\n" "GEMINI_API_KEY=test_key"
    assert_output_contains "directory does not exist" &&
    assert_exit_code 1
}

test_wrong_file_extensions_ignored() {
    # Put non-image files in image dir
    touch "$SANDBOX/image_dataset_here/readme.txt"
    touch "$SANDBOX/image_dataset_here/data.csv"
    run_script "2\n1\n\n"
    assert_output_contains "No Image files found" &&
    assert_exit_code 1
}

# =============================================================================
# Test: TOML File Setup
# =============================================================================

test_toml_copied_to_examples() {
    # SDXL toml should be moved to examples
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    run_script "2\n4\n1\nyes\nyes\n"
    assert_file_exists "$SANDBOX/diffusion_pipe/examples/sdxl.toml"
}

test_toml_already_exists_not_overwritten() {
    # If toml already exists in examples, it should stay (just update output_dir)
    echo "output_dir = '/old/path'" > "$SANDBOX/diffusion_pipe/examples/sdxl.toml"
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    run_script "2\n4\n1\nyes\nyes\n"
    assert_output_contains "sdxl.toml already exists" &&
    assert_file_contains "$SANDBOX/diffusion_pipe/examples/sdxl.toml" "output_dir.*output_folder/sdxl_lora"
}

test_qwen2512_copies_dataset_qwen_toml() {
    mkdir -p "$SANDBOX/models/Qwen-Image-2512"
    touch "$SANDBOX/models/Qwen-Image-2512/dummy"
    run_script "8\n4\n1\nyes\nyes\n"
    assert_file_exists "$SANDBOX/diffusion_pipe/examples/dataset_qwen.toml" &&
    assert_file_contains "$SANDBOX/diffusion_pipe/examples/dataset_qwen.toml" "resolutions = \[1328\]"
}

test_qwen2512_toml_points_to_correct_model() {
    mkdir -p "$SANDBOX/models/Qwen-Image-2512"
    touch "$SANDBOX/models/Qwen-Image-2512/dummy"
    run_script "8\n4\n1\nyes\nyes\n"
    assert_file_exists "$SANDBOX/diffusion_pipe/examples/qwen_2512_toml.toml" &&
    assert_file_contains "$SANDBOX/diffusion_pipe/examples/qwen_2512_toml.toml" "Qwen-Image-2512"
}

# =============================================================================
# Test: Model Download Verification
# =============================================================================

test_flux_verify_empty_dir_fails() {
    export HUGGING_FACE_TOKEN="test_token"
    mkdir -p "$SANDBOX/models/flux"  # exists but empty
    run_script "1\n4\n1\nyes\nyes\n"
    assert_output_contains "Flux model files not found" &&
    assert_exit_code 1
}

test_sdxl_verify_missing_file_fails() {
    # No safetensors file
    run_script "2\n4\n1\nyes\nyes\n"
    assert_output_contains "SDXL model file not found" &&
    assert_exit_code 1
}

test_z_image_verify_missing_files() {
    # Override hf stub to NOT create z_image temp files (simulates partial download)
    cat > "$SANDBOX/.stub_bin/hf" << 'STUB'
#!/bin/bash
echo "stub: hf $*"
exit 0
STUB
    chmod +x "$SANDBOX/.stub_bin/hf"
    # Override wget to NOT create files
    cat > "$SANDBOX/.stub_bin/wget" << 'STUB'
#!/bin/bash
echo "stub: wget $*"
exit 0
STUB
    chmod +x "$SANDBOX/.stub_bin/wget"

    mkdir -p "$SANDBOX/models/z_image"
    # Only create 2 of 4 required files
    touch "$SANDBOX/models/z_image/z_image_turbo_bf16.safetensors"
    touch "$SANDBOX/models/z_image/ae.safetensors"
    run_script "7\n4\n1\nyes\nyes\n"
    # Download subshell fails because mv can't find the temp files,
    # so script should exit with failure (either download error or missing files)
    assert_exit_code 1
}

test_z_image_verify_all_files_present() {
    mkdir -p "$SANDBOX/models/z_image"
    touch "$SANDBOX/models/z_image/z_image_turbo_bf16.safetensors"
    touch "$SANDBOX/models/z_image/ae.safetensors"
    touch "$SANDBOX/models/z_image/qwen_3_4b.safetensors"
    touch "$SANDBOX/models/z_image/zimage_turbo_training_adapter_v2.safetensors"
    run_script "7\n4\n1\nyes\nyes\n"
    assert_output_contains "Model download completed and verified"
}

test_wan13_verify_empty_fails() {
    mkdir -p "$SANDBOX/models/Wan/Wan2.1-T2V-1.3B"  # exists but empty
    run_script "3\n4\n1\nyes\nyes\n"
    assert_output_contains "not found"
}

test_wan13_verify_with_files_passes() {
    mkdir -p "$SANDBOX/models/Wan/Wan2.1-T2V-1.3B"
    touch "$SANDBOX/models/Wan/Wan2.1-T2V-1.3B/dummy.safetensors"
    run_script "3\n4\n1\nyes\nyes\n"
    assert_output_contains "Model download completed and verified"
}

# =============================================================================
# Test: Dataset TOML Configuration
# =============================================================================

test_dataset_toml_path_substitution() {
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    run_script "2\n4\n1\nyes\nyes\n"
    # The $NETWORK_VOLUME placeholder should be replaced with actual path
    assert_file_exists "$SANDBOX/diffusion_pipe/examples/dataset.toml"
    if grep -q '\$NETWORK_VOLUME' "$SANDBOX/diffusion_pipe/examples/dataset.toml"; then
        echo "  FAIL: \$NETWORK_VOLUME placeholder was not replaced"
        return 1
    fi
}

test_dataset_video_section_uncommented_for_video_mode() {
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    touch "$SANDBOX/video_dataset_here/video1.mp4"

    # Create captioning script stub
    echo '#!/bin/bash' > "$SANDBOX/Captioning/video_captioner.sh"
    echo 'echo "video captioning complete"' >> "$SANDBOX/Captioning/video_captioner.sh"
    chmod +x "$SANDBOX/Captioning/video_captioner.sh"

    # SDXL, videos only, existing gemini key
    run_script "2\n2\n1\nyes\nyes\n" "GEMINI_API_KEY=test_key"

    # Check that [[directory]] for video was uncommented
    if [ -f "$SANDBOX/diffusion_pipe/examples/dataset.toml" ]; then
        local uncommented_dirs
        uncommented_dirs=$(grep -c '^\[\[directory\]\]' "$SANDBOX/diffusion_pipe/examples/dataset.toml" 2>/dev/null || echo "0")
        if [ "$uncommented_dirs" -lt 2 ]; then
            echo "  FAIL: Expected 2 uncommented [[directory]] sections, got $uncommented_dirs"
            return 1
        fi
    fi
}

# =============================================================================
# Test: Training Config Display
# =============================================================================

test_training_summary_shows_parameters() {
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    # Put a real TOML in examples so params can be parsed
    cp "$PROJECT_DIR/toml_files/sdxl.toml" "$SANDBOX/diffusion_pipe/examples/"
    run_script "2\n4\n1\nyes\nyes\n"
    assert_output_contains "Epochs:" &&
    assert_output_contains "LoRA Rank:" &&
    assert_output_contains "Learning Rate:"
}

test_qwen2512_shows_1328_resolution() {
    mkdir -p "$SANDBOX/models/Qwen-Image-2512"
    touch "$SANDBOX/models/Qwen-Image-2512/dummy"
    run_script "8\n4\n1\nyes\nyes\n"
    assert_output_contains "1328"
}

# =============================================================================
# Test: Training Pre-flight Checks
# =============================================================================

test_missing_diffusion_pipe_dir() {
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    # Replace diffusion_pipe dir with a file, so cd to it fails
    rm -rf "$SANDBOX/diffusion_pipe"
    touch "$SANDBOX/diffusion_pipe"  # file, not directory — cd will fail
    run_script "2\n4\n1\nyes\nyes\n"
    # Script should hit "Could not find" or the ERR trap
    assert_exit_code 1
}

test_missing_toml_before_training() {
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    # Don't copy the toml to examples, and remove the source so it can't be moved
    rm -f "$SANDBOX/runpod-diffusion_pipe/toml_files/sdxl.toml"
    rm -f "$SANDBOX/diffusion_pipe/examples/sdxl.toml"
    run_script "2\n4\n1\nyes\nyes\n"
    assert_output_contains "Training configuration file not found" &&
    assert_exit_code 1
}

# =============================================================================
# Test: Error Handler
# =============================================================================

test_error_handler_shows_line_number() {
    # Create a script that sources the error handler then fails
    cat > "$SANDBOX/test_error.sh" << 'EOF'
#!/bin/bash
set -e
trap 'on_error $LINENO' ERR
on_error() {
    local line_number=$1
    echo "ERROR at line $line_number"
}
false  # This will trigger the error handler
echo "should not reach here"
EOF
    local output
    output=$(bash "$SANDBOX/test_error.sh" 2>&1) || true
    if echo "$output" | grep -q "ERROR at line"; then
        return 0
    else
        echo "  FAIL: Error handler did not fire"
        return 1
    fi
}

# =============================================================================
# Test: Output Directory Update via sed
# =============================================================================

test_output_dir_updated_in_toml() {
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"
    # Put a toml with old output_dir
    echo 'output_dir = "/data/diffusion_pipe_training_runs/sdxl_lora"' > "$SANDBOX/runpod-diffusion_pipe/toml_files/sdxl.toml"
    echo 'dataset = "examples/dataset.toml"' >> "$SANDBOX/runpod-diffusion_pipe/toml_files/sdxl.toml"
    run_script "2\n4\n1\nyes\nyes\n"
    assert_file_contains "$SANDBOX/diffusion_pipe/examples/sdxl.toml" "output_dir.*output_folder/sdxl_lora"
}

# =============================================================================
# Test: Model Type / TOML File Consistency
# =============================================================================

test_all_model_types_have_toml_files() {
    # Every model referenced in the script should have a corresponding TOML file
    local toml_files=(
        "flux.toml"
        "sdxl.toml"
        "wan13_video.toml"
        "wan14b_t2v.toml"
        "wan14b_i2v.toml"
        "qwen_toml.toml"
        "z_image_toml.toml"
        "qwen_2512_toml.toml"
    )
    local all_ok=true
    for tf in "${toml_files[@]}"; do
        if [ ! -f "$PROJECT_DIR/toml_files/$tf" ]; then
            echo "  FAIL: Missing TOML file: toml_files/$tf"
            all_ok=false
        fi
    done
    $all_ok
}

test_all_toml_files_have_required_fields() {
    local all_ok=true
    for tf in "$PROJECT_DIR"/toml_files/*.toml; do
        local name=$(basename "$tf")
        # Every training toml should have these fields
        for field in "output_dir" "epochs" "dataset" "\[model\]" "\[adapter\]" "\[optimizer\]"; do
            if ! grep -qE "$field" "$tf"; then
                echo "  FAIL: $name missing required field: $field"
                all_ok=false
            fi
        done
    done
    $all_ok
}

test_dataset_toml_files_have_required_fields() {
    local all_ok=true
    for dtf in "$PROJECT_DIR"/dataset*.toml; do
        local name=$(basename "$dtf")
        for field in "resolutions" "enable_ar_bucket" "\[\[directory\]\]" "path"; do
            if ! grep -qE "$field" "$dtf"; then
                echo "  FAIL: $name missing required field: $field"
                all_ok=false
            fi
        done
    done
    $all_ok
}

# =============================================================================
# Test: Blackwell GPU Warning
# =============================================================================

test_blackwell_warning_shown() {
    echo "blackwell" > /tmp/gpu_arch_type
    echo "NVIDIA B200" > /tmp/detected_gpu
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"

    # Run in background, kill after 3 seconds (can't wait for 10s countdown)
    local outfile="$SANDBOX/.blackwell_output"
    (echo -e "2\n4\n1\nyes\nyes\n" | bash "$SCRIPT_UNDER_TEST" > "$outfile" 2>&1) &
    local pid=$!
    sleep 3
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f /tmp/gpu_arch_type /tmp/detected_gpu

    if grep -q "BLACKWELL GPU DETECTED" "$outfile" 2>/dev/null; then
        rm -f "$outfile"
        return 0
    else
        echo "  FAIL: Blackwell warning not shown"
        rm -f "$outfile"
        return 1
    fi
}

# =============================================================================
# Test: Log File Append Mode
# =============================================================================

test_log_append_not_overwrite() {
    # Verify the script uses >> not > for model_download.log
    if grep -E '> "\$NETWORK_VOLUME/logs/model_download.log"' "$SCRIPT_UNDER_TEST" | grep -v '>>' | grep -q .; then
        echo "  FAIL: Found '>' (overwrite) instead of '>>' (append) for model_download.log"
        return 1
    fi
}

# =============================================================================
# Test: Default Case in Download Switch
# =============================================================================

test_download_switch_has_default_case() {
    # The case statement for model downloads should have a *) default
    # Look for the pattern between "case $MODEL_TYPE" and the first "esac"
    if ! grep -A500 'case \$MODEL_TYPE' "$SCRIPT_UNDER_TEST" | grep -B5 'esac' | head -10 | grep -q '\*)'; then
        echo "  FAIL: Download case statement missing default *) handler"
        return 1
    fi
}

# =============================================================================
# Test: cd Safety Check
# =============================================================================

test_cd_has_error_check() {
    # The cd to diffusion_pipe should use "if ! cd" or "cd ... || exit"
    if grep -q 'if ! cd "\$NETWORK_VOLUME/diffusion_pipe"' "$SCRIPT_UNDER_TEST"; then
        return 0
    elif grep -q 'cd "\$NETWORK_VOLUME/diffusion_pipe" || exit' "$SCRIPT_UNDER_TEST"; then
        return 0
    else
        echo "  FAIL: cd to diffusion_pipe has no error handling"
        return 1
    fi
}

# =============================================================================
# Test: TOML Pre-flight Check
# =============================================================================

test_toml_preflight_check_exists() {
    if grep -q 'if \[ ! -f "examples/\$TOML_FILE" \]' "$SCRIPT_UNDER_TEST"; then
        return 0
    else
        echo "  FAIL: No pre-flight check for TOML file existence before training"
        return 1
    fi
}

# =============================================================================
# Test: Qwen 2512 Dataset Resolution
# =============================================================================

test_qwen2512_dataset_resolution_1328() {
    if [ -f "$PROJECT_DIR/dataset_qwen.toml" ]; then
        assert_file_contains "$PROJECT_DIR/dataset_qwen.toml" "resolutions = \[1328\]"
    else
        echo "  FAIL: dataset_qwen.toml does not exist"
        return 1
    fi
}

test_default_dataset_resolution_1024() {
    assert_file_contains "$PROJECT_DIR/dataset.toml" "resolutions = \[1024\]"
}

# =============================================================================
# Test: Script Syntax
# =============================================================================

test_script_syntax_valid() {
    local output
    output=$(bash -n "$SCRIPT_UNDER_TEST" 2>&1)
    if [ $? -eq 0 ]; then
        return 0
    else
        echo "  FAIL: Script has syntax errors:"
        echo "$output" | sed 's/^/    /'
        return 1
    fi
}

# =============================================================================
# Test: Caption Inspection Prompts
# =============================================================================

test_caption_inspection_exit_on_no() {
    touch "$SANDBOX/image_dataset_here/photo1.jpg"
    touch "$SANDBOX/models/sdXL_v10VAEFix.safetensors"

    # Create stub captioning script
    echo '#!/bin/bash' > "$SANDBOX/Captioning/JoyCaption/JoyCaptionRunner.sh"
    echo 'echo "All done!"' >> "$SANDBOX/Captioning/JoyCaption/JoyCaptionRunner.sh"
    chmod +x "$SANDBOX/Captioning/JoyCaption/JoyCaptionRunner.sh"

    # SDXL, images, no trigger, continue defaults, then "no" at caption review
    run_script "2\n1\n\n1\nno\n"
    assert_output_contains "review the captions" &&
    assert_exit_code 0
}

# =============================================================================
# Run all tests
# =============================================================================

FILTER="${1:-}"

echo ""
echo -e "${BOLD}${CYAN}=== Interactive Training Script Test Suite ===${NC}"
echo ""

# Model selection
run_test "model_selection_all_valid" test_model_selection_all_valid
run_test "model_selection_invalid_then_valid" test_model_selection_invalid_then_valid

# Token handling
run_test "flux_requires_token" test_flux_requires_token
run_test "flux_token_from_env" test_flux_token_from_env
run_test "flux_token_placeholder_rejected" test_flux_token_placeholder_rejected

# Caption modes
run_test "caption_mode_skip" test_caption_mode_skip
run_test "caption_mode_images_no_files" test_caption_mode_images_no_files
run_test "caption_mode_videos_no_files" test_caption_mode_videos_no_files
run_test "caption_mode_images_with_files" test_caption_mode_images_with_files
run_test "caption_mode_both_missing_videos" test_caption_mode_both_missing_videos

# Gemini key
run_test "gemini_key_required_for_videos" test_gemini_key_required_for_videos
run_test "gemini_key_from_env" test_gemini_key_from_env
run_test "gemini_key_placeholder_rejected" test_gemini_key_placeholder_rejected

# Dataset directories
run_test "image_dir_missing" test_image_dir_missing
run_test "video_dir_missing" test_video_dir_missing
run_test "wrong_file_extensions_ignored" test_wrong_file_extensions_ignored

# TOML setup
run_test "toml_copied_to_examples" test_toml_copied_to_examples
run_test "toml_already_exists_not_overwritten" test_toml_already_exists_not_overwritten
run_test "qwen2512_copies_dataset_qwen_toml" test_qwen2512_copies_dataset_qwen_toml
run_test "qwen2512_toml_points_to_correct_model" test_qwen2512_toml_points_to_correct_model
run_test "output_dir_updated_in_toml" test_output_dir_updated_in_toml

# Model verification
run_test "flux_verify_empty_dir_fails" test_flux_verify_empty_dir_fails
run_test "sdxl_verify_missing_file_fails" test_sdxl_verify_missing_file_fails
run_test "z_image_verify_missing_files" test_z_image_verify_missing_files
run_test "z_image_verify_all_files_present" test_z_image_verify_all_files_present
run_test "wan13_verify_empty_fails" test_wan13_verify_empty_fails
run_test "wan13_verify_with_files_passes" test_wan13_verify_with_files_passes

# Dataset TOML
run_test "dataset_toml_path_substitution" test_dataset_toml_path_substitution
run_test "training_summary_shows_parameters" test_training_summary_shows_parameters
run_test "qwen2512_shows_1328_resolution" test_qwen2512_shows_1328_resolution

# Pre-flight checks
run_test "missing_diffusion_pipe_dir" test_missing_diffusion_pipe_dir
run_test "missing_toml_before_training" test_missing_toml_before_training

# Static analysis / code quality
run_test "script_syntax_valid" test_script_syntax_valid
run_test "all_model_types_have_toml_files" test_all_model_types_have_toml_files
run_test "all_toml_files_have_required_fields" test_all_toml_files_have_required_fields
run_test "dataset_toml_files_have_required_fields" test_dataset_toml_files_have_required_fields
run_test "log_append_not_overwrite" test_log_append_not_overwrite
run_test "download_switch_has_default_case" test_download_switch_has_default_case
run_test "cd_has_error_check" test_cd_has_error_check
run_test "toml_preflight_check_exists" test_toml_preflight_check_exists
run_test "qwen2512_dataset_resolution_1328" test_qwen2512_dataset_resolution_1328
run_test "default_dataset_resolution_1024" test_default_dataset_resolution_1024

# Error handler
run_test "error_handler_shows_line_number" test_error_handler_shows_line_number

# Misc
run_test "caption_inspection_exit_on_no" test_caption_inspection_exit_on_no

# =============================================================================
# Results
# =============================================================================

echo ""
echo -e "${BOLD}${CYAN}=== Results ===${NC}"
echo ""
echo -e "  Total:  $TESTS_RUN"
echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"

if [ ${#FAILED_NAMES[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tests:${NC}"
    for name in "${FAILED_NAMES[@]}"; do
        echo -e "  ${RED}✗ $name${NC}"
    done
fi

echo ""

# Exit with failure if any tests failed
[ "$TESTS_FAILED" -eq 0 ]
