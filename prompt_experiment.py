import cv2
import numpy as np
import torch
import os
import glob
import sys
from PIL import Image

# Ensure sam3 is importable
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    if os.path.exists("sam3"):
        sys.path.append("sam3")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    else:
        raise ImportError("Could not import sam3.")

# Configuration
IMAGE_DIR = "images"
CHECKPOINT_PATH = "sam3.pt"

TEST_PROMPTS = [
    "pentagon",
    "red pentagon",
    "blue pentagon",
    "pentagonal target",
    "red target",
    "blue target",
    "geometric shape",
    "red geometric shape",
    "blue geometric shape",
    "polygon",
    "pentagonal sign",
    "red pentagonal sign",
    "target board"
]

def main():
    print("Initializing SAM3 for Prompt Experiment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bpe_path = os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    try:
        model = build_sam3_image_model(
            device=device,
            checkpoint_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None,
            load_from_HF=True,
            bpe_path=bpe_path
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    processor = Sam3Processor(model)
    
    img_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not img_files:
        print("No images found.")
        return

    # Just pick the first image for experiment
    img_path = img_files[0] 
    print(f"Testing prompts on: {os.path.basename(img_path)}")
    
    try:
        pil_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    inference_state = processor.set_image(pil_image)
    
    print("\nResults:")
    print(f"{'Prompt':<30} | {'Masks Found':<10}")
    print("-" * 45)
    
    for prompt in TEST_PROMPTS:
        processor.reset_all_prompts(inference_state)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        masks = output.get("masks")
        count = 0
        if masks is not None:
            if hasattr(masks, 'shape'):
                 # masks shape is [N, H, W] or [N, 1, H, W]
                count = masks.shape[0] if masks.shape[0] > 0 else 0
                # Double check if empty mask tensor
                if masks.numel() == 0: count = 0
            elif isinstance(masks, list):
                count = len(masks)
        
        print(f"{prompt:<30} | {count:<10}")

if __name__ == "__main__":
    main()
