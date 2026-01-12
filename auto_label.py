import cv2
import numpy as np
import torch
import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import sys

# === Configuration & Setup ===
# Ensure sam3 is in path
sys.path.insert(0, os.path.join(os.getcwd(), "sam3"))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print("Error: Could not import SAM 3 modules. Make sure 'sam3' folder is in current directory.\n")
    print(f"Details: {e}")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="SAM 3 General Auto Labeling (YOLO BBox)")
    parser.add_argument("--image_dir", type=str, default="images", help="Directory containing images")
    parser.add_argument("--prompt", type=str, default="target", help="Text prompt for detection (e.g., 'car', 'person', 'red target')")
    parser.add_argument("--class_id", type=int, default=0, help="Class ID for YOLO output")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_vis", action="store_true", default=True, help="Save visualization")
    return parser.parse_args()

def mask_to_bbox_yolo(mask, img_w, img_h):
    """
    Convert binary mask to YOLO bbox format: (x_center, y_center, w, h) normalized.
    """
    y, x = np.where(mask)
    if len(x) == 0: return None
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Box dimensions
    w_pixel = x_max - x_min
    h_pixel = y_max - y_min
    
    # Center
    cx_pixel = x_min + w_pixel / 2
    cy_pixel = y_min + h_pixel / 2
    
    # Normalize
    return (cx_pixel / img_w, cy_pixel / img_h, w_pixel / img_w, h_pixel / img_h), (x_min, y_min, x_max, y_max)

def main():
    args = get_args()
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist.")
        return

    vis_dir = "output_auto_label_vis"
    if args.output_vis:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading SAM 3 Model (Device: {args.device})...")
    # Path handling for BPE
    bpe_path = os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        bpe_path = os.path.join("sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    try:
        model = build_sam3_image_model(
            device=args.device,
            checkpoint_path="sam3.pt",
            bpe_path=bpe_path
        )
        processor = Sam3Processor(model, device=args.device, confidence_threshold=args.conf_thresh)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    image_paths = glob.glob(os.path.join(args.image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(args.image_dir, "*.png"))
    
    print(f"Found {len(image_paths)} images. Processing with prompt: '{args.prompt}'")
    
    for img_path in tqdm(image_paths):
        try:
            image_pil = Image.open(img_path).convert("RGB")
            w, h = image_pil.size
            
            # Inference
            inference_state = processor.set_image(image_pil)
            inference_state = processor.set_text_prompt(args.prompt, inference_state)
            
            masks = inference_state["masks"]
            scores = inference_state["scores"]
            
            yolo_lines = []
            vis_img = np.array(image_pil)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            
            if masks is not None and len(masks) > 0:
                for i, mask_tensor in enumerate(masks):
                    score = scores[i].item()
                    if score < args.conf_thresh:
                        continue
                    
                    mask_np = mask_tensor.squeeze().cpu().numpy()
                    
                    res = mask_to_bbox_yolo(mask_np, w, h)
                    if res:
                        (ncx, ncy, nw, nh), (x1, y1, x2, y2) = res
                        
                        # YOLO Format: class_id x_center y_center width height
                        yolo_lines.append(f"{args.class_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                        
                        if args.output_vis:
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save Label
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
            
            # Save Vis
            if args.output_vis and yolo_lines:
                vis_out_path = os.path.join(vis_dir, os.path.basename(img_path))
                cv2.imwrite(vis_out_path, vis_img)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("Done!")

if __name__ == "__main__":
    main()
