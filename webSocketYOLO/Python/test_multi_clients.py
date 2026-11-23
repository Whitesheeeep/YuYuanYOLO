"""
多客户端测试脚本.

功能：模拟多个客户端同时连接到服务器，测试服务器端的客户端管理功能

测试内容：
1. 多个客户端同时连接
2. 每个客户端发送不同的图像
3. 验证服务器只发送消息给对应的客户端
4. 验证监控界面显示所有客户端信息
"""

import asyncio
import base64
import json
import sys

import cv2
import numpy as np
import websockets

# 设置输出编码为 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


async def test_client(device_id, device_name, color, server_url="ws://192.168.5.45:5000"):
    """模拟单个客户端.

    参数：
        device_id: 设备 ID
        device_name: 设备名称
        color: 测试图像的颜色 (B, G, R)
        server_url: 服务器地址
    """
    print(f"[{device_name}] 正在连接到服务器: {server_url}")

    try:
        async with websockets.connect(server_url) as websocket:
            print(f"[{device_name}] 连接成功！")

            # 创建测试图像（不同颜色）
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[:] = color

            # 在图像上添加文字
            cv2.putText(test_image, device_name, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # 编码图像
            _, buffer = cv2.imencode(".jpg", test_image)
            image_base64 = base64.b64encode(buffer).decode()

            # 发送 3 次检测请求
            for i in range(3):
                # 构建测试消息
                message = {"type": "detect", "device_id": device_id, "device_name": device_name, "image": image_base64}

                print(f"[{device_name}] 发送第 {i + 1} 次检测请求...")
                await websocket.send(json.dumps(message))

                # 等待响应
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                data = json.loads(response)

                # 验证响应
                if data["device_id"] == device_id:
                    print(f"[{device_name}] ✓ 收到正确的响应 (检测数量: {len(data.get('detections', []))})")
                else:
                    print(f"[{device_name}] ✗ 收到错误的响应 (device_id: {data['device_id']})")

                # 等待 2 秒再发送下一次
                await asyncio.sleep(2)

            print(f"[{device_name}] 测试完成，保持连接 10 秒...")
            await asyncio.sleep(10)

    except asyncio.TimeoutError:
        print(f"[{device_name}] ✗ 等待响应超时")
    except websockets.exceptions.WebSocketException as e:
        print(f"[{device_name}] ✗ WebSocket 错误: {e}")
    except ConnectionRefusedError:
        print(f"[{device_name}] ✗ 连接被拒绝")
    except Exception as e:
        print(f"[{device_name}] ✗ 未知错误: {type(e).__name__}: {e}")


async def main():
    """主函数：并发运行多个客户端."""
    print("=" * 60)
    print("多客户端测试脚本")
    print("=" * 60)
    print()

    # 创建 3 个不同的客户端
    clients = [
        ("device_001", "测试设备 1", (100, 150, 200)),  # 橙色
        ("device_002", "测试设备 2", (200, 100, 150)),  # 紫色
        ("device_003", "测试设备 3", (150, 200, 100)),  # 青色
    ]

    # 并发运行所有客户端
    tasks = [test_client(device_id, device_name, color) for device_id, device_name, color in clients]

    await asyncio.gather(*tasks)

    print()
    print("=" * 60)
    print("所有客户端测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
