if pgrep -f "huggingface-cli" > /dev/null; then
    echo "Hugging Face CLI download in progress"
    exit 1
fi

cd /

FILE_PATH="$NETWORK_VOLUME/models/sdXL_v10VAEFix.safetensors"

if [ -f "$FILE_PATH" ]; then
	echo "Checkpoint found, starting training"
    cd /diffusion_pipe
	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/sdxl.toml
else
    echo "Checkpoint doesn't exists, exiting"
	exit 1
fi