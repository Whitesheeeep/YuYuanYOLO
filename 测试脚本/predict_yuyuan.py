"""
YuYuan 模型推理脚本
使用训练好的模型进行预测.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


def predict_image(model_path, image_path, save=True):
    """使用训练好的模型预测单张图片.

    Args:
        model_path: 模型权重路径
        image_path: 图片路径
        save: 是否保存结果
    """
    # 加载模型
    model = YOLO(model_path)

    # 预测
    results = model(image_path)

    # 显示结果
    for result in results:
        # 打印检测结果
        print(f"\n检测到 {len(result.boxes)} 个目标:")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names[cls]
            print(f"  - {name}: {conf:.2f}")

        # 绘制结果
        img_with_boxes = result.plot()

        # 转换 BGR 到 RGB
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        # 显示
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"检测结果: {image_path}")
        plt.show()

        # 保存结果
        if save:
            save_path = image_path.replace(".jpg", "_result.jpg")
            cv2.imwrite(save_path, img_with_boxes)
            print(f"\n结果已保存到: {save_path}")


def predict_folder(model_path, folder_path):
    """批量预测文件夹中的图片.

    Args:
        model_path: 模型权重路径
        folder_path: 图片文件夹路径
    """
    model = YOLO(model_path)

    # 批量预测
    results = model(folder_path, save=True, project="runs/predict", name="yuyuan_predict")

    print(f"\n批量预测完成！结果保存在: {results[0].save_dir}")


def letterbox_114(image_rgb, target_size=640, color=(114, 114, 114)):
    """Resize with letterbox padding (Ultralytics default 114)."""
    h, w = image_rgb.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def prepare_input_nchw(image_path, input_size=640):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    letterboxed, scale, pad_x, pad_y = letterbox_114(rgb, input_size)

    inp = letterboxed.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    return inp, scale, pad_x, pad_y, (w, h)


def debug_onnx_alignment(onnx_path, image_path, input_size=640, sample_count=10):
    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime 未安装，请先安装: pip install onnxruntime")
        raise

    inp, scale, pad_x, pad_y, (w, h) = prepare_input_nchw(image_path, input_size)
    flat = inp.ravel()
    sample = ",".join([f"{v:.6f}" for v in flat[:sample_count]])
    print(f"Source size={w}x{h}")
    print(
        f"Input shape={inp.shape} min={flat.min():.6f} max={flat.max():.6f} first={sample} scale={scale:.6f} pad=({pad_x},{pad_y})"
    )

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: inp})
    output = outputs[0]

    if output.ndim != 3:
        print(f"Output shape={output.shape} (unexpected rank)")
        return output

    channels_first = output.shape[1] <= output.shape[2]
    feature_count = output.shape[1] if channels_first else output.shape[2]
    det0 = output[0, :, 0] if channels_first else output[0, 0, :]

    first_det = ",".join([f"{v:.6f}" for v in det0[: min(sample_count, feature_count)]])
    print(
        f"Output shape={output.shape} channelsFirst={channels_first} featureCount={feature_count} firstDet0={first_det}"
    )

    return output


def compare_result_images(python_image_path, unity_image_path, title="Python vs Unity"):
    """Side-by-side visualization of two rendered result images."""
    py_img = cv2.imread(python_image_path)
    unity_img = cv2.imread(unity_image_path)

    if py_img is None:
        raise FileNotFoundError(f"Python result image not found: {python_image_path}")
    if unity_img is None:
        raise FileNotFoundError(f"Unity result image not found: {unity_image_path}")

    py_rgb = cv2.cvtColor(py_img, cv2.COLOR_BGR2RGB)
    unity_rgb = cv2.cvtColor(unity_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(py_rgb)
    axes[0].set_title("Python Result")
    axes[0].axis("off")

    axes[1].imshow(unity_rgb)
    axes[1].set_title("Unity Result")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 使用示例

    # 1. 预测单张图片
    model_path = "runs/train/yuyuan_exp/weights/best.pt"  # 训练好的模型
    image_path = r"/TestImgs/img.png"  # 测试图片路径

    # predict_image(model_path, image_path)

    # 2. 批量预测
    # predict_folder(model_path, 'datasets/YuYuan/images/val')

    # 3. 对齐调试（ONNX 输出）
    onnx_path = r"E:\Master\ultralytics-main\Unity_Integration\Models\best.onnx"
    debug_onnx_alignment(onnx_path, image_path)

    # 4. 结果可视化对比（传入 Python 结果图 & Unity 截图）
    # python_result = r'E:\Master\ultralytics-main\TestImgs\img_result.jpg'
    # unity_result = r'E:\Master\ultralytics-main\TestImgs\img_unity.jpg'
    # compare_result_images(python_result, unity_result)

    print("请取消注释上面的代码来运行预测")
