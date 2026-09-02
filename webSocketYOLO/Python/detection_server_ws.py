import asyncio
import base64
import json

import cv2
import numpy as np
import websockets

from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO(r"E:\Master\ultralytics-main\runs\detect\runs\train\yuyuan_exp\weights\best.pt")
ip = "0.0.0.0"
port = 5000

# 存储所有连接的客户端
clients = set()


async def handle_client(websocket):
    """处理客户端连接."""
    # 添加到客户端列表
    clients.add(websocket)
    print(f"新客户端连接，当前连接数: {len(clients)}")

    try:
        async for message in websocket:
            # 解析消息
            data = json.loads(message)

            if data["type"] == "detect":
                # 解码图像
                image_data = base64.b64decode(data["image"])
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # YOLO 检测
                results = model(img)

                # 绘制检测框
                detected_img = img.copy()
                detections = []

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # 获取坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = model.names[cls]

                        # 绘制边界框
                        cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(
                            detected_img,
                            f"{class_name} {conf:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                        # 添加到检测列表
                        detections.append(
                            {"class": class_name, "confidence": conf, "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                        )

                # 编码图像为 base64
                _, original_buffer = cv2.imencode(".jpg", img)
                original_base64 = base64.b64encode(original_buffer).decode()

                _, detected_buffer = cv2.imencode(".jpg", detected_img)
                detected_base64 = base64.b64encode(detected_buffer).decode()

                # 构建响应
                response = {
                    "type": "detection_result",
                    "device_id": data["device_id"],
                    "device_name": data["device_name"],
                    "original_image": original_base64,
                    "detected_image": detected_base64,
                    "detections": detections,
                }

                # 广播到所有客户端（包括监控界面）
                await broadcast(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # 移除客户端
        clients.remove(websocket)
        print(f"客户端断开，当前连接数: {len(clients)}")


async def broadcast(message):
    """广播消息到所有客户端."""
    if clients:
        await asyncio.gather(*[client.send(message) for client in clients], return_exceptions=True)


async def main():
    print("启动 WebSocket 服务器...")
    print(f"监听地址: ws://{ip}:{port}")

    async with websockets.serve(handle_client, ip, port):
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    asyncio.run(main())
