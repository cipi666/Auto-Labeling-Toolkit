import cv2
import numpy as np
import torch
import os
import glob
import sys
os.environ['CUDA_VISIBLE_DEVICES']='2' 
# === 路径配置 ===
SAM2_REPO_PATH = "."  # 指向 sam2 源码目录
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml" 
CHECKPOINT = "sam2.1_hiera_tiny.pt"
IMAGE_DIR = "/home/pchen/ngyf2024/project/images"
OUTPUT_DIR = "output_hybrid"
DEVICE = "cpu"  # 您的环境暂时用CPU，Tiny模型很快

# === 引入 SAM2 ===
# 确保能导入 sam2
if SAM2_REPO_PATH not in sys.path:
    sys.path.append(SAM2_REPO_PATH)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= 1. OpenCV 粗定位 (只找点，不找轮廓) =================
def get_prompts_from_color(image):
    """
    返回:
    prompts: list of dict, [{'point': [x,y], 'label': class_id}, ...]
    vis_img: 用于调试的中间图
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    prompts = []
    
    # --- 红色阈值 (HSV中红色在0和180两头) ---
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_r1 + mask_r2
    
    # --- 蓝色阈值 ---
    lower_blue = np.array([100, 80, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 查找红色中心点
    contours_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_r:
        area = cv2.contourArea(cnt)
        if area > 50: # 过滤太小的噪点
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                prompts.append({'point': [cx, cy], 'label': 0}) # 0: Red

    # 查找蓝色中心点
    contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_b:
        area = cv2.contourArea(cnt)
        if area > 50:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                prompts.append({'point': [cx, cy], 'label': 1}) # 1: Blue
                
    return prompts

# ================= 2. 工具函数 =================
def mask_to_yolo(mask, img_w, img_h):
    y, x = np.where(mask)
    if len(x) == 0: return None
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    return (cx/img_w, cy/img_h, w/img_w, h/img_h), (x_min, y_min, x_max, y_max)

def main():
    # Load Model
    cfg_path = os.path.join(SAM2_REPO_PATH, "sam2_configs", MODEL_CFG)
    ckpt_path = os.path.join(SAM2_REPO_PATH, CHECKPOINT)
    
    print("正在加载 SAM 2 Predictor...")
    sam2_model = build_sam2(MODEL_CFG, ckpt_path, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    img_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    for img_path in img_files:
        print(f"处理: {os.path.basename(img_path)}")
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # --- Step 1: OpenCV 找提示点 ---
        prompts = get_prompts_from_color(image)
        if not prompts:
            print("  -> 未发现红/蓝目标点")
            continue
            
        # --- Step 2: 设置 SAM 图像 ---
        predictor.set_image(image_rgb)
        
        txt_lines = []
        vis_img = image.copy()
        
        # --- Step 3: 对每个点进行 Prompt 推理 ---
        for p in prompts:
            point_coords = np.array([p['point']]) # Shape: [1, 2]
            point_labels = np.array([1])          # 1 表示这是一个前景点
            class_id = p['label']
            
            # 让 SAM 预测 Mask
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True # 输出3个mask，我们要选分数最高的
            )
            
            # 取分数最高的 Mask
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            
            # --- Step 4: 转 YOLO ---
            res = mask_to_yolo(best_mask, w, h)
            if res:
                (ycx, ycy, yw, yh), (x1, y1, x2, y2) = res
                txt_lines.append(f"{class_id} {ycx:.6f} {ycy:.6f} {yw:.6f} {yh:.6f}")
                
                # 画图可视化
                color = (0, 0, 255) if class_id == 0 else (255, 0, 0)
                # 画框
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                # 画提示点 (证明是用这个点生成的)
                cv2.circle(vis_img, tuple(p['point']), 5, (0, 255, 255), -1)
                # 画半透明 Mask
                vis_img[best_mask > 0] = vis_img[best_mask > 0] * 0.5 + np.array(color) * 0.5

        # 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 保存 TXT
        with open(os.path.join(IMAGE_DIR, base_name + ".txt"), 'w') as f:
            f.write("\n".join(txt_lines))
            
        # 保存可视化图
        cv2.imwrite(os.path.join(OUTPUT_DIR, base_name + "_vis.jpg"), vis_img)
        print(f"  -> 保存 {len(txt_lines)} 个目标")

if __name__ == "__main__":
    main()