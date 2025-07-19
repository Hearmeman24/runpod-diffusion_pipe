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
1. **Model Selection** - Choose from Flux, SDXL, or Wan models
2. **API Keys** - Enter required tokens (Hugging Face for Flux, Gemini for video captioning)
3. **Dataset Options** - Select image captioning, video captioning, or both
4. **Configuration Review** - Review training parameters before starting

### Step 4: Wait for Training to Complete
The script will automatically:
- Download the selected model
- Generate captions for your media (if selected)
- Start LoRA training with optimized settings

## Training Results

Once training is complete, your trained LoRA files will be saved in:
```
training_outputs
```

## Dataset Preparation

Before running the script, place your training data in:
- **Images**: `image_dataset_here/` folder
- **Videos**: `video_dataset_here/` folder

## Tips

- **First Run**: Allow extra time for model downloads (can be several GB)
- **API Keys**: Have your Hugging Face token ready for Flux, Gemini API key for video captioning
- **Monitor Progress**: The script shows progress indicators for downloads and captioning
- **Review Captions**: You'll be prompted to manually review generated captions before training starts

## Need Help?

The interactive script provides clear instructions and error messages to guide you through each step. Simply follow the on-screen prompts!
