"""
导出 YuYuan YOLO 模型为 ONNX 格式
供 Unity 调用.
"""

import os

from ultralytics import YOLO

# 模型路径
MODEL_PATH = r"E:\Master\ultralytics-main\runs\detect\runs\train\yuyuan_exp\weights\best.pt"


def export_to_onnx():
    """导出模型为 ONNX 格式."""
    print("=" * 60)
    print("YuYuan 模型导出为 ONNX")
    print("=" * 60)

    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print("❌ 错误: 模型文件不存在!")
        print(f"   路径: {MODEL_PATH}")
        return

    print(f"\n加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("\n开始导出为 ONNX 格式...")
    print("这可能需要几分钟时间，请耐心等待...")

    # 导出为 ONNX
    # imgsz: 输入图像大小，Unity 中使用时需要匹配这个尺寸
    # simplify: 简化 ONNX 模型，提高兼容性
    # opset: ONNX opset 版本
    export_path = model.export(
        format="onnx",
        imgsz=640,  # 输入图像大小
        simplify=True,  # 简化模型
        opset=12,  # ONNX opset 版本（Unity Barracuda 推荐 12）
        dynamic=False,  # 固定输入尺寸（Unity 推荐）
    )

    print("\n" + "=" * 60)
    print("✓ 导出成功!")
    print("=" * 60)
    print(f"\nONNX 模型保存在: {export_path}")

    # 获取模型信息
    model_dir = os.path.dirname(export_path)
    model_name = os.path.basename(export_path)

    print("\n模型信息:")
    print(f"  - 文件名: {model_name}")
    print(f"  - 目录: {model_dir}")
    print("  - 输入尺寸: 640x640")
    print("  - 类别数: 5")
    print("  - 类别: Stone1, Picture, LionLeft, LionRight, Stone2")

    print("\n在 Unity 中使用:")
    print(f"  1. 将 {model_name} 导入到 Unity 项目的 Assets 文件夹")
    print("  2. 使用 Unity Barracuda 加载模型")
    print("  3. 输入图像需要预处理为 640x640，归一化到 [0,1]")
    print("  4. 输出包含检测框、置信度和类别信息")

    return export_path


if __name__ == "__main__":
    export_to_onnx()
