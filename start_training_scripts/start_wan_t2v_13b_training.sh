if pgrep -f "huggingface-cli" > /dev/null; then
    echo "Hugging Face CLI download in progress"
    exit 1
fi

cd /

FILE_PATH="$NETWORK_VOLUME/Wan/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"

# This check is stupid and I know it, I'll fix it in the future :) 

if [ -f "$FILE_PATH" ]; then
	echo "Wan model found, starting training"
    cd /diffusion_pipe
	NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan13_video.toml
else
    echo "Model doesn't exists, exiting"
	exit 1
fi

