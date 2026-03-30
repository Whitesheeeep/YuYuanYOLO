"""
YuYuan 数据集验证脚本
验证训练好的模型性能
"""
from ultralytics import YOLO

def validate_model(model_path, data_yaml):
    """
    验证模型在验证集上的性能

    Args:
        model_path: 模型权重路径
        data_yaml: 数据集配置文件路径
    """
    # 加载模型
    model = YOLO(model_path)

    # 在验证集上评估
    metrics = model.val(data=data_yaml)

    # 打印结果
    print("\n=== 验证结果 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    # 每个类别的性能
    print("\n=== 各类别性能 ===")
    for i, name in enumerate(model.names.values()):
        print(f"{name}: mAP50={metrics.box.maps[i]:.4f}")

    return metrics

def export_model(model_path, format='onnx'):
    """
    导出模型为其他格式

    Args:
        model_path: 模型权重路径
        format: 导出格式 ('onnx', 'torchscript', 'tflite', 等)
    """
    model = YOLO(model_path)
    model.export(format=format)
    print(f"\n模型已导出为 {format} 格式")

if __name__ == '__main__':
    # 验证模型
    model_path = r'E:\Master\ultralytics-main\Unity_Integration\Models\best.onnx'
    data_yaml = 'datasets/YuYuan/data.yaml'

    validate_model(model_path, data_yaml)

    # 导出模型（可选）
    # export_model(model_path, format='onnx')

    print("请取消注释上面的代码来运行验证")
