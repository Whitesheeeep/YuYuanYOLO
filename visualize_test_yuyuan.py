"""
YuYuan 可视化测试脚本
显示检测结果和边界框
"""
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np

# 模型路径
MODEL_PATH = r'E:\Master\ultralytics-main\runs\detect\runs\train\yuyuan_exp\weights\best.pt'
VAL_IMAGES = r'E:\Master\ultralytics-main\datasets\YuYuan\images\val'

# 类别颜色映射
COLORS = {
    'Stone1': '#FF6B6B',      # 红色
    'Picture': '#4ECDC4',     # 青色
    'LionLeft': '#45B7D1',    # 蓝色
    'LionRight': '#FFA07A',   # 橙色
    'Stone2': '#98D8C8'       # 绿色
}

def visualize_single_image(model, image_path, conf_threshold=0.25):
    """
    可视化单张图片的检测结果

    Args:
        model: YOLO 模型
        image_path: 图片路径
        conf_threshold: 置信度阈值
    """
    # 预测
    results = model(image_path, conf=conf_threshold)
    result = results[0]

    # 读取原图
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 左图：原图 + 边界框（手动绘制）
    ax1.imshow(img_rgb)
    ax1.set_title(f'检测结果 - {Path(image_path).name}\n检测到 {len(result.boxes)} 个目标',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 绘制边界框和标签
    for box in result.boxes:
        # 获取坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cls]

        # 获取颜色
        color = COLORS.get(name, '#FFFFFF')

        # 绘制矩形框
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=3,
            edgecolor=color,
            facecolor='none'
        )
        ax1.add_patch(rect)

        # 绘制标签背景
        label = f'{name} {conf:.2f}'
        text_bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8)
        ax1.text(x1, y1-10, label,
                fontsize=12,
                color='white',
                fontweight='bold',
                bbox=text_bbox)

    # 右图：使用 YOLO 自带的绘制方法
    img_with_boxes = result.plot()
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    ax2.imshow(img_with_boxes_rgb)
    ax2.set_title('YOLO 原生绘制', fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()

    # 打印检测信息
    print(f"\n{'='*60}")
    print(f"图片: {Path(image_path).name}")
    print(f"{'='*60}")
    print(f"检测到 {len(result.boxes)} 个目标:")
    for i, box in enumerate(result.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        print(f"  [{i+1}] {name:12s} - 置信度: {conf:.3f} - 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    return fig

def visualize_multiple_images(model, image_folder, num_images=6, conf_threshold=0.25):
    """
    可视化多张图片的检测结果（网格显示）

    Args:
        model: YOLO 模型
        image_folder: 图片文件夹路径
        num_images: 显示图片数量
        conf_threshold: 置信度阈值
    """
    # 获取图片列表
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
    image_files = image_files[:num_images]

    if not image_files:
        print(f"❌ 在 {image_folder} 中没有找到图片")
        return

    # 计算网格大小
    n_cols = 3
    n_rows = (len(image_files) + n_cols - 1) // n_cols

    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    print(f"\n{'='*60}")
    print(f"批量可视化测试 - 共 {len(image_files)} 张图片")
    print(f"{'='*60}")

    # 统计信息
    total_detections = 0
    class_counts = {}

    for idx, image_path in enumerate(image_files):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 预测
        results = model(str(image_path), conf=conf_threshold)
        result = results[0]

        # 显示图片
        img_with_boxes = result.plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f'{image_path.name}\n检测: {len(result.boxes)} 个目标',
                    fontsize=10, fontweight='bold')
        ax.axis('off')

        # 统计
        total_detections += len(result.boxes)
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        print(f"  [{idx+1}] {image_path.name:30s} - 检测到 {len(result.boxes)} 个目标")

    # 隐藏多余的子图
    for idx in range(len(image_files), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    # 打印统计信息
    print(f"\n{'='*60}")
    print("统计信息")
    print(f"{'='*60}")
    print(f"总图片数: {len(image_files)}")
    print(f"总检测数: {total_detections}")
    print(f"平均检测: {total_detections/len(image_files):.2f} 个/图")
    print(f"\n各类别检测数量:")
    for cls_name, count in sorted(class_counts.items()):
        print(f"  {cls_name:12s}: {count:3d} 个")

    return fig

def visualize_with_confidence_comparison(model, image_path):
    """
    对比不同置信度阈值的检测结果

    Args:
        model: YOLO 模型
        image_path: 图片路径
    """
    thresholds = [0.1, 0.25, 0.5, 0.7]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    print(f"\n{'='*60}")
    print(f"置信度阈值对比 - {Path(image_path).name}")
    print(f"{'='*60}")

    for idx, conf in enumerate(thresholds):
        results = model(image_path, conf=conf)
        result = results[0]

        img_with_boxes = result.plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f'置信度阈值: {conf}\n检测到 {len(result.boxes)} 个目标',
                           fontsize=14, fontweight='bold')
        axes[idx].axis('off')

        print(f"  阈值 {conf}: 检测到 {len(result.boxes)} 个目标")
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            print(f"    - {cls_name}: {conf_val:.3f}")

    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("\n" + "="*60)
    print("YuYuan 可视化测试")
    print("="*60)
    print(f"模型: {MODEL_PATH}")

    # 加载模型
    print("\n加载模型...")
    model = YOLO(MODEL_PATH)
    print("✓ 模型加载成功")

    print("\n请选择测试模式:")
    print("1. 单张图片详细可视化")
    print("2. 批量图片网格显示（6张）")
    print("3. 批量图片网格显示（12张）")
    print("4. 置信度阈值对比")
    print("5. 测试验证集所有图片")

    choice = input("\n请输入选项 (1-5): ").strip()

    if choice == '1':
        image_path = input("请输入图片路径（或按回车使用验证集第一张）: ").strip()
        if not image_path:
            val_images = list(Path(VAL_IMAGES).glob('*.jpg'))
            if val_images:
                image_path = str(val_images[0])
            else:
                print("❌ 验证集中没有图片")
                return
        visualize_single_image(model, image_path)
        plt.show()

    elif choice == '2':
        visualize_multiple_images(model, VAL_IMAGES, num_images=6)
        plt.show()

    elif choice == '3':
        visualize_multiple_images(model, VAL_IMAGES, num_images=12)
        plt.show()

    elif choice == '4':
        image_path = input("请输入图片路径（或按回车使用验证集第一张）: ").strip()
        if not image_path:
            val_images = list(Path(VAL_IMAGES).glob('*.jpg'))
            if val_images:
                image_path = str(val_images[0])
            else:
                print("❌ 验证集中没有图片")
                return
        visualize_with_confidence_comparison(model, image_path)
        plt.show()

    elif choice == '5':
        visualize_multiple_images(model, VAL_IMAGES, num_images=100)
        plt.show()

    else:
        print("无效选项")

if __name__ == '__main__':
    main()
