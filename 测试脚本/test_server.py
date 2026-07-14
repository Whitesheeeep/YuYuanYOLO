"""
测试检测服务器
快速验证服务器是否正常工作.
"""

import base64
import json
from pathlib import Path

import requests

# 服务器地址
SERVER_URL = "http://localhost:5000"


def test_health():
    """测试健康检查接口."""
    print("\n" + "=" * 60)
    print("测试 1: 健康检查")
    print("=" * 60)

    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_detect(image_path):
    """测试检测接口."""
    print("\n" + "=" * 60)
    print("测试 2: 图像检测")
    print("=" * 60)

    if not Path(image_path).exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return False

    try:
        # 读取并编码图像
        print(f"读取图像: {image_path}")
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        print(f"图像大小: {len(image_data)} 字符")

        # 发送请求
        print("发送检测请求...")
        response = requests.post(f"{SERVER_URL}/detect", json={"image": image_data, "confidence": 0.25}, timeout=30)

        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n✓ 检测成功!")
            print(f"  - 检测数量: {result['count']}")
            print(f"  - 推理时间: {result['inference_time'] * 1000:.1f}ms")
            print(f"  - 图像尺寸: {result['image_size']['width']}x{result['image_size']['height']}")

            if result["detections"]:
                print("\n检测结果:")
                for i, det in enumerate(result["detections"]):
                    print(f"  [{i + 1}] {det['class_name']}")
                    print(f"      置信度: {det['confidence']:.3f}")
                    bbox = det["bbox"]
                    print(f"      位置: ({bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f})")
            else:
                print("\n未检测到目标")

            return True
        else:
            print(f"❌ 检测失败: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("YuYuan 检测服务器测试")
    print("=" * 60)
    print(f"服务器地址: {SERVER_URL}")

    # 测试 1: 健康检查
    health_ok = test_health()

    if not health_ok:
        print("\n❌ 服务器未启动或无法连接")
        print("请先运行: python detection_server.py")
        return

    # 测试 2: 图像检测
    # 使用项目中的测试图像
    test_image = r"E:\Master\ultralytics-main\bus.jpg"

    if not Path(test_image).exists():
        print(f"\n⚠️ 测试图像不存在: {test_image}")
        print("请指定一个有效的图像路径")
        test_image = input("输入图像路径: ").strip()

    if Path(test_image).exists():
        detect_ok = test_detect(test_image)

        if detect_ok:
            print("\n" + "=" * 60)
            print("✓ 所有测试通过!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ 检测测试失败")
            print("=" * 60)
    else:
        print("\n❌ 无法找到测试图像")


if __name__ == "__main__":
    main()
