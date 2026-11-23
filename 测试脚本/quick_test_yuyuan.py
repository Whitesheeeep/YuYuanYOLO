"""快速测试脚本 - 直接测试验证集."""

import os

from ultralytics import YOLO

# 模型和数据路径
MODEL_PATH = r"/runs/detect/runs/train/yuyuan_exp/weights/best.pt"
VAL_IMAGES = r"E:\Master\ultralytics-main\datasets\YuYuan\images\val"


def quick_test():
    """快速测试验证集."""
    print("=" * 60)
    print("YuYuan 模型快速测试")
    print("=" * 60)

    # 检查路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        return

    if not os.path.exists(VAL_IMAGES):
        print(f"❌ 验证集不存在: {VAL_IMAGES}")
        return

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 批量预测验证集
    print(f"\n开始测试验证集: {VAL_IMAGES}")
    results = model(
        VAL_IMAGES,
        conf=0.25,  # 置信度阈值
        save=True,  # 保存结果图片
        save_txt=True,  # 保存检测结果文本
        save_conf=True,  # 保存置信度
        project="runs/test",  # 保存目录
        name="yuyuan_quick_test",
        exist_ok=True,
    )

    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果统计")
    print("=" * 60)

    total_images = len(results)
    total_detections = sum(len(r.boxes) for r in results)

    # 统计每个类别的检测数量
    class_counts = {}
    for result in results:
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    print(f"\n总图片数: {total_images}")
    print(f"总检测数: {total_detections}")
    print(f"平均每张图片检测: {total_detections / total_images:.2f} 个目标")

    print("\n各类别检测数量:")
    for cls_name, count in sorted(class_counts.items()):
        print(f"  {cls_name:12s}: {count:3d} 个")

    print("\n✓ 结果已保存到: runs/test/yuyuan_quick_test")
    print("  - 检测结果图片")
    print("  - 检测结果文本 (.txt)")


if __name__ == "__main__":
    quick_test()
