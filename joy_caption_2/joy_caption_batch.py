#!/usr/bin/env python3
"""
Joy Caption Batch Processing Script
Processes all images in a directory and generates caption files using JoyCaption model.
"""

import os
import argparse
import logging
from pathlib import Path
from PIL import Image
import torch
import gc
import threading
from typing import Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('joy_caption_batch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class JoyCaptionManager:
    def __init__(self, timeout_minutes: int = 5):
        self.model = None
        self.processor = None
        # Proper device detection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.timeout = timeout_minutes * 60
        self.timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()
        self.model_name = "fancyfeast/llama-joycaption-beta-one-hf-llava"

    def load_model(self):
        with self.lock:
            if self.model is None:
                logger.info("Loading JoyCaption model...")
                try:
                    from transformers import AutoProcessor, LlavaForConditionalGeneration

                    # Load processor
                    logger.info("Loading processor...")
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )

                    # Load model
                    logger.info("Loading model...")
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )

                    # Fix missing pad_token
                    tok = self.processor.tokenizer
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                        self.model.config.pad_token_id = tok.eos_token_id

                    # Move model to device if not using device_map="auto"
                    if self.device != "cuda":
                        self.model = self.model.to(self.device)

                    logger.info(f"Model loaded successfully on {self.device}")

                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise

    def unload_model(self):
        with self.lock:
            if self.model is not None:
                logger.info("Unloading model from memory...")
                del self.model
                del self.processor
                self.model = None
                self.processor = None

                # Clear cache based on device
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                gc.collect()
                logger.info("Model unloaded successfully")

    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self.unload_model)
        self.timer.start()

    def generate_caption(self, image: Image.Image, prompt: str = "Write a descriptive caption for this image.") -> str:
        self.load_model()
        self.reset_timer()

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            logging.info(f"Original prompt: {prompt}")

            # Determine image token
            image_token = getattr(self.processor, 'image_token',
                                  getattr(self.processor.tokenizer, 'image_token', '<image>'))
            logging.info(f"Image token: {image_token}")

            prompt_text = f"{image_token}\n{prompt}"
            logging.info(f"Formatted prompt: {prompt_text}")

            inputs = self.processor(
                images=image,
                text=prompt_text,
                return_tensors="pt",
                padding=True
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            logging.info("Generating caption...")
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    top_k=0,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            input_len = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0][input_len:]
            caption = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

            logger.info(f"Generated caption: {caption}")
            return caption

        except Exception as e:
            logger.error(f"Error in generate_caption: {e}")
            raise


def get_image_files(directory: Path) -> list:
    """Get all image files from the directory."""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []

    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)

    return sorted(image_files)


def process_images(input_dir: str, output_dir: str = None, prompt: str = "Write a descriptive caption for this image.",
                   skip_existing: bool = True, timeout_minutes: int = 5, trigger_word: str = None):
    """
    Process all images in the input directory and generate captions.

    Args:
        input_dir: Directory containing images to process
        output_dir: Directory to save caption files (defaults to same as input_dir)
        prompt: Caption generation prompt
        skip_existing: Skip images that already have caption files
        timeout_minutes: Model unload timeout in minutes
        trigger_word: Optional trigger word to prepend to generated captions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = get_image_files(input_path)

    if not image_files:
        logger.warning(f"No image files found in {input_path}")
        return

    logger.info(f"Found {len(image_files)} image files to process")

    # Initialize caption manager
    caption_manager = JoyCaptionManager(timeout_minutes=timeout_minutes)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    try:
        for i, image_file in enumerate(image_files, 1):
            try:
                # Generate caption file path
                caption_file = output_path / f"{image_file.stem}.txt"

                # Skip if caption file already exists and skip_existing is True
                if skip_existing and caption_file.exists():
                    logger.info(f"[{i}/{len(image_files)}] Skipping {image_file.name} - caption file already exists")
                    skipped_count += 1
                    continue

                logger.info(f"[{i}/{len(image_files)}] Processing {image_file.name}")

                # Load and process image
                with Image.open(image_file) as img:
                    caption = caption_manager.generate_caption(img, prompt)

                # Add trigger word if specified
                if trigger_word:
                    caption = f"{trigger_word}, {caption}"

                # Save caption to file
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)

                logger.info(f"[{i}/{len(image_files)}] Saved caption to {caption_file.name}")
                processed_count += 1

            except Exception as e:
                logger.error(f"[{i}/{len(image_files)}] Error processing {image_file.name}: {e}")
                error_count += 1
                continue

    finally:
        # Ensure model is unloaded
        logger.info("Unloading model...")
        caption_manager.unload_model()

        # Cancel any pending timer
        if caption_manager.timer:
            caption_manager.timer.cancel()

    # Print summary
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total images found: {len(image_files)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already exists): {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Batch process images with JoyCaption')
    parser.add_argument('input_dir', help='Directory containing images to process')
    parser.add_argument('--output-dir', help='Directory to save caption files (defaults to input directory)')
    parser.add_argument('--prompt', default='Write a descriptive caption for this image.',
                        help='Caption generation prompt')
    parser.add_argument('--trigger-word',
                        help='Trigger word to prepend to generated captions (e.g., "claude" -> "claude, <caption>")')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Process all images even if caption files already exist')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Model unload timeout in minutes (default: 5)')

    args = parser.parse_args()

    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        skip_existing=not args.no_skip_existing,
        timeout_minutes=args.timeout,
        trigger_word=args.trigger_word
    )


if __name__ == "__main__":
    main()