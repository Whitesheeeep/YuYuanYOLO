"""
YuYuan 模型测试脚本
测试训练好的 YOLO 模型
"""
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 模型路径
MODEL_PATH = r'/runs/detect/runs/train/yuyuan_exp/weights/best.pt'

def test_single_image(image_path, conf_threshold=0.25):
    """
    测试单张图片

    Args:
        image_path: 图片路径
        conf_threshold: 置信度阈值
    """
    print(f"\n{'='*50}")
    print(f"测试图片: {image_path}")
    print(f"{'='*50}")

    # 加载模型
    model = YOLO(MODEL_PATH)

    # 预测
    results = model(image_path, conf=conf_threshold)

    # 显示检测结果
    result = results[0]
    print(f"\n检测到 {len(result.boxes)} 个目标:")

    for i, box in enumerate(result.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = result.names[cls]
        xyxy = box.xyxy[0].cpu().numpy()

        print(f"  [{i+1}] {name}")
        print(f"      置信度: {conf:.3f}")
        print(f"      位置: x1={xyxy[0]:.1f}, y1={xyxy[1]:.1f}, x2={xyxy[2]:.1f}, y2={xyxy[3]:.1f}")

    # 绘制结果
    img_with_boxes = result.plot()
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # 显示
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'检测结果 (共 {len(result.boxes)} 个目标)', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 保存结果
    save_path = str(Path(image_path).parent / f"{Path(image_path).stem}_result.jpg")
    cv2.imwrite(save_path, img_with_boxes)
    print(f"\n✓ 结果已保存到: {save_path}")

    return results

def test_validation_set():
    """
    在验证集上测试模型性能
    """
    print(f"\n{'='*50}")
    print("在验证集上评估模型")
    print(f"{'='*50}")

    model = YOLO(MODEL_PATH)

    # 验证
    metrics = model.val(data=r'E:\Master\ultralytics-main\datasets\YuYuan\data.yaml')

    print("\n=== 整体性能 ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\n=== 各类别性能 ===")
    for i, name in enumerate(model.names.values()):
        print(f"{name:12s}: mAP50={metrics.box.maps[i]:.4f}")

    return metrics

def test_batch_images(folder_path, conf_threshold=0.25):
    """
    批量测试文件夹中的图片

    Args:
        folder_path: 图片文件夹路径
        conf_threshold: 置信度阈值
    """
    print(f"\n{'='*50}")
    print(f"批量测试: {folder_path}")
    print(f"{'='*50}")

    model = YOLO(MODEL_PATH)

    # 批量预测
    results = model(
        folder_path,
        conf=conf_threshold,
        save=True,
        project='runs/test',
        name='yuyuan_test'
    )

    # 统计结果
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\n✓ 测试完成!")
    print(f"  - 测试图片数: {len(results)}")
    print(f"  - 检测目标数: {total_detections}")
    print(f"  - 结果保存在: runs/test/yuyuan_test")

    return results

def test_with_different_thresholds(image_path):
    """
    使用不同置信度阈值测试

    Args:
        image_path: 图片路径
    """
    print(f"\n{'='*50}")
    print("测试不同置信度阈值")
    print(f"{'='*50}")

    model = YOLO(MODEL_PATH)
    thresholds = [0.1, 0.25, 0.5, 0.7]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, conf in enumerate(thresholds):
        results = model(image_path, conf=conf)
        result = results[0]

        img_with_boxes = result.plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'置信度阈值: {conf} (检测到 {len(result.boxes)} 个目标)', fontsize=12)

        print(f"置信度 {conf}: 检测到 {len(result.boxes)} 个目标")

    plt.tight_layout()
    plt.show()

def main():
    """
    主测试函数
    """
    print("\n" + "="*50)
    print("YuYuan 模型测试")
    print("="*50)
    print(f"模型路径: {MODEL_PATH}")

    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 错误: 模型文件不存在!")
        print(f"   请检查路径: {MODEL_PATH}")
        return

    print("\n请选择测试模式:")
    print("1. 测试单张图片")
    print("2. 在验证集上评估")
    print("3. 批量测试文件夹")
    print("4. 测试不同置信度阈值")
    print("5. 运行所有测试")

    choice = input("\n请输入选项 (1-5): ").strip()

    if choice == '1':
        image_path = input("请输入图片路径: ").strip()
        test_single_image(image_path)

    elif choice == '2':
        test_validation_set()

    elif choice == '3':
        folder_path = input("请输入文件夹路径: ").strip()
        test_batch_images(folder_path)

    elif choice == '4':
        image_path = input("请输入图片路径: ").strip()
        test_with_different_thresholds(image_path)

    elif choice == '5':
        # 运行所有测试
        print("\n>>> 1. 验证集评估")
        test_validation_set()

        print("\n>>> 2. 批量测试验证集图片")
        val_folder = r'E:\Master\ultralytics-main\datasets\YuYuan\images\val'
        if os.path.exists(val_folder):
            test_batch_images(val_folder)
        else:
            print(f"验证集文件夹不存在: {val_folder}")

    else:
        print("无效选项!")

if __name__ == '__main__':
    main()
