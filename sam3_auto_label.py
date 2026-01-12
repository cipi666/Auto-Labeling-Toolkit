import cv2
import numpy as np
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='2' 
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import sys

# === 配置 ===
# 如果 sam3 文件夹在当前目录下，添加到路径
sys.path.insert(0, os.path.join(os.getcwd(), "sam3"))

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print("错误: 无法导入 SAM 3 模块。请确保 'sam3' 文件夹在当前目录下或已安装。\n")
    print(f"详细错误: {e}")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="SAM 3 Auto Labeling Tool with Geometric Ordering")
    parser.add_argument("--image_dir", type=str, default="images", help="图片文件夹路径")
    parser.add_argument("--prompt", type=str, default="pentagonal sign", help="用于检测的文本提示 (例如 'pentagonal sign', 'target')")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--output_vis", action="store_true", default=True, help="是否保存可视化结果")
    return parser.parse_args()

def mask_to_polygon(mask, epsilon_factor=0.02):
    """
    将二进制掩码转换为多边形，尝试拟合为 5 个点。
    """
    # 取最大的轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 取最大的轮廓
    cnt = max(contours, key=cv2.contourArea)
    
    # 凸包 (五边形通常是凸的)
    hull = cv2.convexHull(cnt)
    
    # 多边形拟合
    epsilon = epsilon_factor * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # 如果点数不是5，尝试调整 epsilon
    # 简单的自适应逻辑
    if len(approx) != 5:
        for factor in [0.01, 0.03, 0.04, 0.05, 0.005]:
            epsilon = factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 5:
                break
    
    # 如果还是不行，强制采样5个点（不仅是不准确，但保证格式）
    # 或者如果点数 > 5，取最显著的5个点（这里简单处理：如果不为5，则丢弃或保留原样）
    # 为保证数据质量，如果不是5边形，我们暂时认为这个物体检测不合格，返回 None
    # 实际工程中可能需要更复杂的逻辑 (比如 K-Means 聚类轮廓点)
    if len(approx) != 5:
        # Fallback: 如果点数接近5 (比如4或6)，可能只是角点没找对
        # 这里为了演示，如果不是5个点，我们返回 None (宁缺毋滥)
        # print(f"Warning: Found polygon with {len(approx)} points, skipping.")
        return None
        
    return approx.reshape(-1, 2)

def main():
    args = get_args()
    
    # 1. 准备目录
    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录 '{args.image_dir}' 不存在\n")
        return
        
    vis_dir = "output_sam3_vis"
    if args.output_vis:
        os.makedirs(vis_dir, exist_ok=True)
        
    # 2. 加载模型
    print(f"正在加载 SAM 3 模型 (Device: {args.device})...")
    # 假设权重文件在根目录 sam3.pt
    bpe_path = os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        # 尝试备用路径
        bpe_path = os.path.join("sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
        
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path="sam3.pt", # 默认根目录
        bpe_path=bpe_path
    )
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.conf_thresh)
    
    # 3. 处理图片
    image_paths = glob.glob(os.path.join(args.image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(args.image_dir, "*.png"))
    
    print(f"找到 {len(image_paths)} 张图片，开始处理...")
    
    for img_path in tqdm(image_paths):
        try:
            image_pil = Image.open(img_path).convert("RGB")
            w, h = image_pil.size
            
            # --- SAM 3 推理 ---
            # 1. 设置图片
            inference_state = processor.set_image(image_pil)
            # 2. 文本提示推理
            inference_state = processor.set_text_prompt(args.prompt, inference_state)
            
            # 获取结果
            masks = inference_state["masks"] # [N, H, W] Bool Tensor
            scores = inference_state["scores"]
            
            yolo_lines = []
            vis_img = np.array(image_pil)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            
            for i, mask_tensor in enumerate(masks):
                score = scores[i].item()
                if score < args.conf_thresh:
                    continue
                
                mask_np = mask_tensor.squeeze().cpu().numpy() # [H, W] bool
                
                # --- 多边形处理 ---
                pts = mask_to_polygon(mask_np)
                
                if pts is not None:
                    # --- 转换为 YOLO Segmentation 格式 ---
                    # <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn> (归一化 0-1)
                    # 假设 pentagon class_id 为 0
                    line_parts = ["0"]
                    for pt in pts:
                        nx = pt[0] / w
                        ny = pt[1] / h
                        line_parts.append(f"{nx:.6f} {ny:.6f}")
                    yolo_lines.append(" ".join(line_parts))
                    
                    # --- 可视化 ---
                    if args.output_vis:
                        # 画轮廓
                        cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
                        # 标记所有点并标序号
                        for idx, pt in enumerate(pts):
                            cv2.circle(vis_img, tuple(pt), 4, (255, 0, 0), -1)
                            cv2.putText(vis_img, str(idx), (pt[0]+5, pt[1]-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 保存 Label
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
                
            # 保存可视化
            if args.output_vis and yolo_lines:
                vis_out_path = os.path.join(vis_dir, os.path.basename(img_path))
                cv2.imwrite(vis_out_path, vis_img)
                
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}\n")
            continue

    print("处理完成！")

if __name__ == "__main__":
    main()
