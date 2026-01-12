# SAM 3 自动标注工具箱 (SAM 3 Auto-Labeling Toolkit)

本项目利用 **SAM 3 (Segment Anything Model 3)** 实现高精度的图像自动化标注。特别针对需要后续透视变换（摆正）的靶标识别任务进行了优化。

## 🎯 核心特性：几何感知排序

为了支持后续的数字识别和姿态校正，本工具生成的标注文件具有严格的**几何顺序**：

1.  **自动五边形拟合**：将分割 Mask 自动拟合为精确的 5 个顶点。
2.  **固定顶点顺序**：
    *   **关键点 0 (Index 0)**：始终是五边形的**“顶点”**（算法默认取 Y 轴坐标最小的点，即视觉上的最高点）。
    *   **后续点 (Index 1-4)**：严格按照**顺时针**方向排列。

这一特性使得下游的 YOLO 模型能够学习到顶点的概念，从而在推理阶段直接输出有序的关键点，方便直接进行 `cv2.getPerspectiveTransform`。

## 📂 项目结构

```
.
├── sam3_auto_label.py       # 核心脚本：文本提示 -> 有序 YOLO 标签
├── sam3.pt                  # 模型权重文件 (需放在此处)
├── sam3/                    # SAM 3 源码库
├── images/                  # 输入图片文件夹 (输出的 .txt 标签也会保存在这里)
└── output_sam3_vis/         # 可视化结果 (用于检查标注质量)
```

## 🚀 快速开始

### 1. 环境准备
确保已安装 PyTorch, OpenCV, Pillow 等基础库，以及项目目录下的 `sam3` 依赖。

```bash
pip install torch torchvision opencv-python pillow tqdm
cd sam3 && pip install -e . && cd ..
```

### 2. 准备数据
将待标注的图片（.jpg/.png）放入 `images/` 文件夹。

### 3. 运行自动标注
运行脚本，它会自动下载模型（如果配置了 HF Token）或加载本地 `sam3.pt`。

```bash
# 基本用法
python sam3_auto_label.py

# 自定义参数
python sam3_auto_label.py \
  --image_dir my_dataset \
  --prompt "pentagonal sign" \
  --conf_thresh 0.5 \
  --device cuda
```

### 4. 检查结果
*   **标签文件**：在 `images/` 下生成的同名 `.txt` 文件。
    *   格式：`0 <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <x5> <y5>`
    *   其中 `(x1, y1)` 保证是顶点。
*   **可视化**：查看 `output_sam3_vis/` 文件夹。
    *   **红色大点**：顶点 (Index 0)。
    *   **蓝色小点**：其余点，按顺时针带数字标记。

## 📊 下游任务建议 (YOLO)

使用此工具生成的数据训练 YOLO-Pose 或 YOLO-Segment 模型后，推理时的后处理流程建议：

1.  **获取关键点**：模型输出的 Shape 为 `(5, 2)`。
2.  **计算变换矩阵**：
    ```python
    # 假设 pred_kpts 是模型预测出的5个点
    src_pts = pred_kpts.astype(np.float32)
    
    # 定义目标形状（例如正五边形或标准矩形，取决于你想怎么摆正）
    # 这里以摆正为"正五边形"为例，或者只利用前三个点做仿射变换
    # ...
    
    # 简单的摆正逻辑：利用 Index 0 (头) 和 Index 2,3 (底) 计算角度
    top_point = src_pts[0]
    # ...
    ```

## ⚠️ 注意事项
*   脚本假设大部分靶标在图像中是相对直立的（用于确定初始顶点）。如果您的数集包含大量倒置（180度旋转）的靶标，可能需要手动检查或修改 `order_points_pentagon` 中的逻辑。
