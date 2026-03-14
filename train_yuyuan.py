"""
YuYuan 数据集训练脚本
训练 YOLOv8 模型识别：Stone1, Picture, LionLeft, LionRight, Stone2.
"""

import torch

from ultralytics import YOLO


def main():
    # 检查 CUDA 是否可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载预训练模型
    model = YOLO("yolov8n.pt")  # 使用 nano 模型（最快）
    # model = YOLO('yolov8s.pt')  # 或使用 small 模型（更准确）
    # model = YOLO('yolov8m.pt')  # 或使用 medium 模型（平衡）

    # 训练参数
    results = model.train(
        data="datasets/YuYuan/data.yaml",  # 数据集配置文件
        epochs=100,  # 训练轮数
        imgsz=640,  # 图像大小
        batch=16,  # 批次大小（根据显存调整）
        device=device,  # 使用的设备
        workers=4,  # 数据加载线程数
        project="runs/train",  # 保存路径
        name="yuyuan_exp",  # 实验名称
        exist_ok=True,  # 允许覆盖已存在的实验
        # 优化参数
        patience=50,  # 早停耐心值
        save=True,  # 保存检查点
        save_period=10,  # 每 10 轮保存一次
        # 数据增强
        hsv_h=0.015,  # 色调增强
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        degrees=0.0,  # 旋转角度
        translate=0.1,  # 平移
        scale=0.5,  # 缩放
        flipud=0.0,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        mosaic=1.0,  # Mosaic 增强
    )

    print("\n训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")


if __name__ == "__main__":
    main()
