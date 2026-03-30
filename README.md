# HearmemanAI LoRA Trainer - Quick Start Guide

Deploy here: https://get.runpod.io/diffusion-pipe-template

## Getting Started

### Step 1: Open Terminal
Click the **Terminal** button to open a command prompt.

### Step 2: Start the Interactive Training Script
Type the following command and press Enter:
```bash
bash interactive_start_training.sh
```

### Step 3: Follow the Interactive Setup
The script will guide you through:
1. **Model Selection** - Choose from 8 supported models (see below)
2. **API Keys** - Enter required tokens (Hugging Face for Flux, Gemini for video captioning)
3. **Dataset Options** - Select image captioning, video captioning, or both
4. **Configuration Review** - Review and optionally modify training parameters before starting

### Step 4: Wait for Training to Complete
The script will automatically:
- Download the selected model (with live progress bar)
- Generate captions for your media (with progress tracking)
- Start LoRA training with optimized settings

## Supported Models

| # | Model | Type | Notes |
|---|-------|------|-------|
| 1 | **Flux** | Image | Requires Hugging Face token |
| 2 | **SDXL** | Image | Stable Diffusion XL |
| 3 | **Wan 1.3B** | Video (T2V) | Lightweight text-to-video |
| 4 | **Wan 14B T2V** | Video (T2V) | Large text-to-video, supports both T2V and I2V |
| 5 | **Wan 14B I2V** | Video (I2V) | Image-to-video, advanced users only |
| 6 | **Qwen Image** | Image | 1024 resolution |
| 7 | **Z Image Turbo** | Image | Multi-component model with turbo adapter |
| 8 | **Qwen Image 2512** | Image | 1328 resolution |

## Training Results

Once training is complete, your trained LoRA files will be saved in:
```
output_folder/
```

## Dataset Preparation

Before running the script, place your training data in:
- **Images**: `image_dataset_here/` folder
- **Videos**: `video_dataset_here/` folder

## Tips

- **First Run**: Allow extra time for model downloads (can be several GB)
- **API Keys**: Have your Hugging Face token ready for Flux, Gemini API key for video captioning
- **Monitor Progress**: The script shows progress bars for downloads and captioning
- **Review Captions**: You'll be prompted to manually review generated captions before training starts
- **Configuration**: You can modify training parameters (epochs, learning rate, LoRA rank, etc.) before training starts

## Need Help?

The interactive script provides clear error messages with troubleshooting hints. If something fails, check the logs in the `logs/` directory.
