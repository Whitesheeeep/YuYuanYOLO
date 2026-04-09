import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import sys
import pytest

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

@pytest.mark.asyncio
async def test_connection():
    """测试 WebSocket 连接"""
    server_url = "ws://192.168.1.110:5000"

    print(f"正在连接到服务器: {server_url}")

    try:
        async with websockets.connect(server_url) as websocket:
            print("[OK] 连接成功！")

            # 创建一个测试图像（纯色图像）
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[:] = (100, 150, 200)  # BGR 颜色

            # 在图像上添加文字
            cv2.putText(test_image, "Test Image", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # 编码为 JPEG
            _, buffer = cv2.imencode('.jpg', test_image)
            image_base64 = base64.b64encode(buffer).decode()

            # 构建测试消息
            message = {
                "type": "detect",
                "device_id": "test_device_001",
                "device_name": "Test Client",
                "image": image_base64
            }

            print("[>>] 发送测试图像...")
            await websocket.send(json.dumps(message))
            print("[OK] 消息已发送")

            # 等待响应
            print("[..] 等待服务器响应...")
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=180.0)

                print("[OK] 收到响应！")
                data = json.loads(response)

                print(f"\n响应内容:")
                print(f"  类型: {data.get('type')}")

                if data.get('type') == 'command':
                    print(f"  [命令] button_id={data.get('button_id')}  button_name={data.get('button_name')}")
                else:
                    print(f"  设备ID: {data.get('device_id')}")
                    print(f"  设备名称: {data.get('device_name')}")
                    print(f"  检测数量: {len(data.get('detections', []))}")

            if data.get('detections'):
                print(f"\n检测到的目标:")
                for i, det in enumerate(data['detections'], 1):
                    print(f"    {i}. {det['class']} (置信度: {det['confidence']:.2f})")

            print("\n[OK] 测试成功！服务器工作正常。")

    except asyncio.TimeoutError:
        print("[ERROR] 等待响应超时")
    except websockets.exceptions.WebSocketException as e:
        print(f"[ERROR] WebSocket 错误: {e}")
    except ConnectionRefusedError:
        print("[ERROR] 连接被拒绝，请确认:")
        print("   1. 服务器是否正在运行")
        print("   2. IP 地址和端口是否正确")
        print("   3. 防火墙是否阻止了连接")
    except Exception as e:
        print(f"[ERROR] 未知错误: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket 连接测试脚本")
    print("=" * 60)
    asyncio.run(test_connection())
