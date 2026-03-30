"""
YuYuan 目标检测服务器
使用 Flask 提供 HTTP API 接口供 Unity 客户端调用
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from PIL import Image
import io
import time

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载模型（启动时加载一次）
MODEL_PATH = r'/runs/detect/runs/train/yuyuan_exp/weights/best.pt'
model = None

# 类别名称
CLASS_NAMES = ['Stone1', 'Picture', 'LionLeft', 'LionRight', 'Stone2']

def load_model():
    """加载 YOLO 模型"""
    global model
    print("正在加载模型...")
    model = YOLO(MODEL_PATH)
    print("✓ 模型加载成功")

def decode_image(image_data):
    """
    解码 Base64 图像数据

    Args:
        image_data: Base64 编码的图像字符串

    Returns:
        numpy array: OpenCV 格式的图像
    """
    # 解码 Base64
    image_bytes = base64.b64decode(image_data)

    # 转换为 PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # 转换为 OpenCV 格式 (BGR)
    image_np = np.array(image)

    # 如果是 RGBA，转换为 RGB
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_np

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    目标检测接口

    请求格式:
    {
        "image": "base64_encoded_image_data",
        "confidence": 0.25  (可选)
    }

    响应格式:
    {
        "success": true,
        "detections": [
            {
                "class_id": 0,
                "class_name": "Stone1",
                "confidence": 0.85,
                "bbox": {
                    "x1": 100, "y1": 200,
                    "x2": 300, "y2": 400
                }
            }
        ],
        "inference_time": 0.032,
        "count": 1
    }
    """
    try:
        start_time = time.time()

        # 检查模型是否加载
        if model is None:
            return jsonify({
                'success': False,
                'error': '模型未加载'
            }), 500

        # 获取请求数据
        data = request.get_json()

        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': '缺少图像数据'
            }), 400

        # 获取置信度阈值（默认 0.25）
        confidence_threshold = data.get('confidence', 0.25)

        # 解码图像
        image = decode_image(data['image'])

        # 运行检测
        results = model(image, conf=confidence_threshold)
        result = results[0]

        # 解析检测结果
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                'class_id': cls_id,
                'class_name': CLASS_NAMES[cls_id],
                'confidence': round(conf, 3),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                }
            })

        inference_time = time.time() - start_time

        # 返回结果
        response = {
            'success': True,
            'detections': detections,
            'count': len(detections),
            'inference_time': round(inference_time, 3),
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        }

        print(f"检测完成: {len(detections)} 个目标, 用时 {inference_time*1000:.1f}ms")

        return jsonify(response)

    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    """
    批量检测接口

    请求格式:
    {
        "images": ["base64_1", "base64_2", ...],
        "confidence": 0.25  (可选)
    }
    """
    try:
        data = request.get_json()

        if 'images' not in data:
            return jsonify({
                'success': False,
                'error': '缺少图像数据'
            }), 400

        confidence_threshold = data.get('confidence', 0.25)
        images_data = data['images']

        # 解码所有图像
        images = [decode_image(img_data) for img_data in images_data]

        # 批量检测
        start_time = time.time()
        results = model(images, conf=confidence_threshold)
        inference_time = time.time() - start_time

        # 解析结果
        all_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    'class_id': cls_id,
                    'class_name': CLASS_NAMES[cls_id],
                    'confidence': round(conf, 3),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                })
            all_detections.append(detections)

        return jsonify({
            'success': True,
            'results': all_detections,
            'total_images': len(images),
            'inference_time': round(inference_time, 3)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 启动时加载模型
    load_model()

    # 启动服务器
    print("\n" + "="*60)
    print("YuYuan 检测服务器启动")
    print("="*60)
    print(f"服务器地址: http://localhost:5000")
    print(f"检测接口: POST http://localhost:5000/detect")
    print(f"健康检查: GET http://localhost:5000/health")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
