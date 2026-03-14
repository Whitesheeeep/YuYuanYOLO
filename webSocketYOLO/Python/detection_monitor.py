"""
YuYuan YOLO 检测监控系统.
======================

功能概述：
1. 作为 WebSocket 服务器，接收来自 Unity 客户端的图像数据
2. 使用 YOLO 模型进行目标检测
3. 通过 PyQt5 界面实时显示检测结果
4. 支持多设备同时连接和监控

架构设计：
- WebSocket 服务器运行在后台线程（asyncio 事件循环）
- PyQt5 界面运行在主线程
- 使用 Qt 信号槽机制实现线程安全的数据传递
- 监控界面通过全局信号直接接收检测结果，无需 WebSocket 客户端连接

数据流向：
Unity 客户端 --WebSocket--> 服务器 --YOLO检测--> 信号发射器 --Qt信号--> 监控界面
                                    └--> 广播结果 --> Unity 客户端
"""

import asyncio
import base64
import hashlib
import json
import socket
import sys
import threading

import cv2
import numpy as np
import websockets
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ultralytics import YOLO

# ============================================================================
# 全局配置和初始化
# ============================================================================


def get_local_ip():
    """获取本机局域网 IP 地址.

    功能：获取本机在局域网中的 IP 地址，供客户端连接使用

    返回：
        str: 本机 IP 地址，如果获取失败返回 "127.0.0.1"

    实现原理：
        - 创建一个 UDP socket 连接到外部地址（不实际发送数据）
        - 通过 getsockname() 获取本机使用的网络接口 IP
        - 这种方法可以自动选择正确的网络接口（如果有多个网卡）
    """
    try:
        # 创建 UDP socket（不会实际发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到外部地址（8.8.8.8 是 Google DNS，这里只是用来确定路由）
        s.connect(("8.8.8.8", 80))
        # 获取本机 IP
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        # 如果获取失败，返回本地回环地址
        return "127.0.0.1"


# 加载 YOLO 模型（程序启动时加载一次，所有检测共享同一个模型实例）
model = YOLO(r"../runs/detect/runs/train/yuyuan_exp/weights/best.pt")

# WebSocket 服务器配置
ip = "0.0.0.0"  # 监听所有网络接口（允许局域网内的设备连接）
port = 5000  # 服务器端口
local_ip = get_local_ip()  # 获取本机局域网 IP

# 存储所有连接的 WebSocket 客户端（Unity 设备）
# 使用字典存储：{device_id: ClientSession}
clients = {}

# 用户可配置的类别颜色（OpenCV BGR 格式）
# 示例：USER_CLASS_COLORS = {"Stone1": (0, 255, 0), "Stone2": (255, 0, 0)}
USER_CLASS_COLORS = {}

# 默认调色板（BGR），当没有显式配置时使用
CLASS_COLOR_PALETTE = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (128, 0, 128),
]


class ClassColorMap:
    """类别颜色映射，支持用户覆盖并提供稳定的默认色。."""

    def __init__(self, user_colors=None, palette=None, default_color=(0, 255, 0)):
        self.user_colors = user_colors or {}
        self.palette = palette or []
        self.default_color = default_color

    def set_color(self, class_name, bgr_color):
        self.user_colors[class_name] = tuple(bgr_color)

    def get(self, class_name, class_index=None):
        if class_name in self.user_colors:
            return tuple(self.user_colors[class_name])
        if class_index is not None and 0 <= class_index < len(self.palette):
            return self.palette[class_index]
        if class_name and self.palette:
            digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % len(self.palette)
            return self.palette[idx]
        return self.default_color


class_color_map = ClassColorMap(USER_CLASS_COLORS, CLASS_COLOR_PALETTE)
# ============================================================================
# 客户端会话管理
# ============================================================================


class ClientSession:
    """客户端会话类.

    功能：存储单个客户端的连接信息和状态

    属性：
        device_id: 设备唯一标识符
        device_name: 设备名称
        websocket: WebSocket 连接对象
        remote_address: 客户端远程地址 (IP, port)
        connected_at: 连接时间戳
    """

    def __init__(self, device_id, device_name, websocket, remote_address):
        self.device_id = device_id
        self.device_name = device_name
        self.websocket = websocket
        self.remote_address = remote_address  # (ip, port) 元组
        self.connected_at = asyncio.get_event_loop().time()

    @property
    def ip(self):
        """获取客户端 IP 地址."""
        return self.remote_address[0] if self.remote_address else "unknown"

    @property
    def port(self):
        """获取客户端端口."""
        return self.remote_address[1] if self.remote_address else 0

    def __repr__(self):
        return (
            f"ClientSession(device_id={self.device_id}, device_name={self.device_name}, ip={self.ip}, port={self.port})"
        )


# ============================================================================
# 线程间通信机制
# ============================================================================


class SignalEmitter(QObject):
    """全局信号发射器.

    作用：实现从 WebSocket 线程（后台线程）向 Qt 主线程传递数据 原理：Qt 的信号槽机制是线程安全的，可以跨线程传递数据

    为什么需要这个：
    - WebSocket 服务器运行在 asyncio 事件循环的后台线程中
    - PyQt5 界面运行在主线程中
    - 不能直接从后台线程操作 Qt 界面组件（会导致崩溃）
    - 通过信号槽机制，后台线程发射信号，主线程接收并更新界面
    """

    detection_result = pyqtSignal(dict)  # 检测结果信号，携带字典类型的数据
    client_connected = pyqtSignal(str, str, str, int)  # 客户端连接信号 (device_id, device_name, ip, port)
    client_disconnected = pyqtSignal(str)  # 客户端断开信号 (device_id)


# 创建全局信号发射器实例（整个程序共享）
signal_emitter = SignalEmitter()

# ============================================================================
# 辅助函数
# ============================================================================


def apply_nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制（NMS）.

    功能：过滤掉重叠度高的检测框，只保留置信度最高的框 原理：
        1. 按置信度从高到低排序
        2. 选择置信度最高的框
        3. 计算该框与其他框的 IoU（交并比）
        4. 删除 IoU 大于阈值的框（重叠度高）
        5. 重复步骤 2-4，直到处理完所有框

    参数：
        boxes: numpy 数组，形状为 (N, 4)，每行为 [x1, y1, x2, y2]
        scores: numpy 数组，形状为 (N,)，每个框的置信度
        iou_threshold: IoU 阈值，默认 0.5

    返回：
        保留的框的索引列表
    """
    if len(boxes) == 0:
        return []

    # 转换为 numpy 数组
    boxes = np.array(boxes)
    scores = np.array(scores)

    # 获取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按置信度排序（从高到低）
    order = scores.argsort()[::-1]

    keep = []  # 保留的框的索引

    while order.size > 0:
        # 选择置信度最高的框
        i = order[0]
        keep.append(i)

        # 计算该框与其他框的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算 IoU（交并比）
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# ============================================================================
# WebSocket 服务器核心逻辑
# ============================================================================


async def handle_client(websocket):
    """处理单个客户端连接.

    功能：
    1. 接收来自 Unity 客户端的图像数据
    2. 使用 YOLO 模型进行目标检测
    3. 绘制检测框和标签
    4. 将结果发送到监控界面和对应的客户端

    参数：
        websocket: WebSocket 连接对象

    消息格式（接收）： {
        "type": "detect",
        "device_id": "device_001",
        "device_name": "Android Device 1",
        "image": "base64_encoded_image_data"
    }

    消息格式（发送）： {
        "type": "detection_result",
        "device_id": "device_001",
        "device_name": "Android Device 1",
        "original_image": "base64_encoded_original",
        "detected_image": "base64_encoded_with_boxes",
        "detections": [
            {
                "class": "Stone1",
                "confidence": 0.85,
                "bbox": [x1, y1, x2, y2]
            }
        ]
    }
    """
    # 获取客户端远程地址（IP 和端口）
    remote_address = websocket.remote_address
    print(f"新连接来自: {remote_address[0]}:{remote_address[1]}")

    # 临时存储 device_id，用于断开连接时清理
    current_device_id = None

    try:
        # 持续监听客户端消息（异步迭代器）
        async for message in websocket:
            # 解析 JSON 消息
            data = json.loads(message)

            # 处理检测请求
            if data["type"] == "detect":
                device_id = data["device_id"]
                device_name = data["device_name"]

                # ========== 步骤 0: 注册或更新客户端会话 ==========
                if device_id not in clients:
                    # 新客户端，创建会话
                    session = ClientSession(device_id, device_name, websocket, remote_address)
                    clients[device_id] = session
                    current_device_id = device_id
                    print(f"[注册] 新设备: {session}")
                    print(f"当前连接设备数: {len(clients)}")

                    # 发送客户端连接信号到监控界面
                    signal_emitter.client_connected.emit(device_id, device_name, session.ip, session.port)
                else:
                    # 已存在的客户端，更新 WebSocket 连接（可能重连）
                    session = clients[device_id]
                    session.websocket = websocket
                    session.remote_address = remote_address
                    current_device_id = device_id

                # ========== 步骤 1: 解码图像 ==========
                # 将 base64 编码的图像数据解码为二进制
                image_data = base64.b64decode(data["image"])
                # 将二进制数据转换为 numpy 数组
                nparr = np.frombuffer(image_data, np.uint8)
                # 使用 OpenCV 解码为图像矩阵（BGR 格式）
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # ========== 步骤 2: YOLO 检测 ==========
                # 调用 YOLO 模型进行目标检测
                # results 包含检测到的所有目标信息（边界框、类别、置信度等）
                results = model(img)

                # ========== 步骤 3: 收集所有检测框并应用 NMS ==========
                # 先收集所有检测框的信息
                all_boxes = []
                all_scores = []
                all_classes = []
                all_class_names = []

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # 提取边界框坐标（xyxy 格式：左上角和右下角坐标）
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 提取置信度（检测的可信程度，0-1 之间）
                        conf = float(box.conf[0])
                        # 提取类别索引
                        cls = int(box.cls[0])
                        # 获取类别名称（如 "Stone1", "Stone2" 等）
                        class_name = model.names[cls]

                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(conf)
                        all_classes.append(cls)
                        all_class_names.append(class_name)

                # 应用 NMS 过滤重叠框（IoU 阈值 0.5）
                if len(all_boxes) > 0:
                    keep_indices = apply_nms(all_boxes, all_scores, iou_threshold=0.5)
                else:
                    keep_indices = []

                # ========== 步骤 4: 绘制过滤后的检测框 ==========
                # 复制原始图像，用于绘制检测结果
                detected_img = img.copy()
                # 存储所有检测结果的列表
                detections = []

                # 只绘制 NMS 保留的框
                for idx in keep_indices:
                    x1, y1, x2, y2 = all_boxes[idx]
                    conf = all_scores[idx]
                    class_name = all_class_names[idx]
                    class_index = all_classes[idx]

                    # 为不同类别选择不同颜色
                    box_color = class_color_map.get(class_name, class_index)

                    # 在图像上绘制边界框
                    cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    # 在边界框上方绘制类别名称和置信度
                    cv2.putText(
                        detected_img,
                        f"{class_name} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )

                    # 将检测结果添加到列表（用于 JSON 响应）
                    detections.append(
                        {"class": class_name, "confidence": conf, "bbox": [int(x1), int(y1), int(x2), int(y2)]}
                    )

                # ========== 步骤 5: 编码图像为 base64 ==========
                # 将原始图像编码为 JPEG 格式
                _, original_buffer = cv2.imencode(".jpg", img)
                # 转换为 base64 字符串（用于网络传输）
                original_base64 = base64.b64encode(original_buffer).decode()

                # 将检测后的图像编码为 JPEG 格式
                _, detected_buffer = cv2.imencode(".jpg", detected_img)
                # 转换为 base64 字符串
                detected_base64 = base64.b64encode(detected_buffer).decode()

                # ========== 步骤 6: 构建响应消息 ==========
                response = {
                    "type": "detection_result",
                    "device_id": device_id,
                    "device_name": device_name,
                    "original_image": original_base64,  # 原始图像
                    "detected_image": detected_base64,  # 带检测框的图像
                    "detections": detections,  # 检测结果列表
                }

                # ========== 步骤 7: 发送结果 ==========
                # 6.1 通过 Qt 信号发送到监控界面（线程安全）
                # 监控界面需要接收所有设备的检测结果
                signal_emitter.detection_result.emit(response)

                # 6.2 只发送给对应的客户端（精确投递，不广播）
                # 服务器端判断消息归属，客户端无需判断
                await send_to_client(device_id, json.dumps(response))

                if len(detections) > 0:
                    print(
                        f"[检测] 设备 {device_name} ({session.ip}:{session.port}) - 检测到 {len(detections)} 个目标 (NMS 过滤后)"
                    )

    except websockets.exceptions.ConnectionClosed:
        # 客户端正常断开连接
        print(f"[断开] 客户端断开连接: {remote_address[0]}:{remote_address[1]}")
    except Exception as e:
        # 其他异常
        print(f"[错误] 处理客户端消息时出错: {e}")
    finally:
        # 清理：从连接字典中移除断开的客户端
        if current_device_id and current_device_id in clients:
            removed_session = clients.pop(current_device_id)
            print(f"[移除] 设备: {removed_session}")
            print(f"当前连接设备数: {len(clients)}")

            # 发送客户端断开信号到监控界面
            signal_emitter.client_disconnected.emit(current_device_id)


async def send_to_client(device_id, message):
    """发送消息给指定的客户端.

    功能：根据 device_id 精确发送消息给对应的客户端 优势：
        - 避免广播造成的带宽浪费
        - 客户端无需判断消息是否属于自己
        - 服务器端统一管理消息路由

    参数：
        device_id: 目标设备 ID
        message: JSON 字符串格式的消息

    异常处理：
        - 如果客户端不存在，记录警告
        - 如果发送失败，捕获异常并记录
    """
    if device_id in clients:
        session = clients[device_id]
        try:
            await session.websocket.send(message)
        except Exception as e:
            print(f"[错误] 发送消息到设备 {device_id} 失败: {e}")
    else:
        print(f"[警告] 设备 {device_id} 不在连接列表中")


async def broadcast(message):
    """广播消息到所有连接的客户端（保留此函数以备将来使用）.

    功能：将消息同时发送给所有连接的设备 使用场景：
        - 系统通知
        - 全局配置更新
        - 服务器状态广播

    参数：
        message: JSON 字符串格式的消息

    实现细节：
        - 使用 asyncio.gather 并发发送，提高效率
        - return_exceptions=True 确保某个客户端发送失败不影响其他客户端
    """
    if clients:
        await asyncio.gather(*[session.websocket.send(message) for session in clients.values()], return_exceptions=True)


async def start_websocket_server():
    """启动 WebSocket 服务器.

    功能：创建 WebSocket 服务器并持续运行 监听地址：ws://{ip}:{port}

    实现细节：
        - 使用 websockets.serve 创建服务器
        - await asyncio.Future() 让服务器永久运行（直到程序退出）
    """
    print("=" * 60)
    print("启动 WebSocket 服务器")
    print("=" * 60)
    print(f"监听地址: ws://{ip}:{port} (所有网络接口)")
    print(f"局域网连接地址: ws://{local_ip}:{port}")
    print(f"本地连接地址: ws://127.0.0.1:{port}")
    print("=" * 60)

    async with websockets.serve(handle_client, ip, port):
        await asyncio.Future()  # 永久运行


def run_websocket_server():
    """在单独线程中运行 WebSocket 服务器.

    作用：将 asyncio 事件循环运行在后台线程中 原因：
        - PyQt5 需要占用主线程运行事件循环
        - WebSocket 服务器需要 asyncio 事件循环
        - 两者不能在同一个线程中运行，因此需要分离

    调用方式：
        server_thread = threading.Thread(target=run_websocket_server, daemon=True)
        server_thread.start()
    """
    asyncio.run(start_websocket_server())


# ============================================================================
# PyQt5 监控界面
# ============================================================================


class MonitorWindow(QMainWindow):
    """监控窗口主类.

    功能：
    1. 显示所有连接的设备列表
    2. 显示选中设备的原始视频流和检测后视频流
    3. 实时更新检测结果
    4. 支持切换不同设备的视频流

    界面布局：
    ┌─────────────────────────────────────────────────────┐
    │  设备列表  │  原始视频流  │  检测后视频流          │
    │  ● 设备1   │              │                        │
    │  ● 设备2   │              │                        │
    │            │              │                        │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YuYuan 检测监控系统")
        self.setGeometry(100, 100, 1400, 900)

        # 存储设备信息：{device_id: {'name': device_name, 'ip': ip, 'port': port}}
        self.devices = {}
        # 当前选中的设备 ID
        self.current_device_id = None

        # 初始化界面
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面.

        布局结构：
        - 主布局：水平分割器（QSplitter）
          - 左侧：设备列表（QListWidget）
          - 右侧：视频显示区域
            - 上方：服务器信息标签
            - 中间：双视频窗口（原始 + 检测后）
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter()

        # ========== 左侧：设备列表 ==========
        self.device_list = QListWidget()
        # 连接点击事件：当用户点击设备时，切换显示该设备的视频流
        self.device_list.itemClicked.connect(self.on_device_selected)
        splitter.addWidget(self.device_list)

        # ========== 右侧：视频显示区域 ==========
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # 服务器信息标签
        self.server_info_label = QLabel()
        self.server_info_label.setAlignment(Qt.AlignCenter)
        self.server_info_label.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; "
            "padding: 10px; border: 2px solid #00ff00; "
            "border-radius: 5px; font-weight: bold;"
        )
        server_info_text = (
            f"🌐 WebSocket 服务器运行中\n局域网地址: ws://{local_ip}:{port}\n本地地址: ws://127.0.0.1:{port}"
        )
        self.server_info_label.setText(server_info_text)
        self.server_info_label.setFont(QFont("Consolas", 10))
        self.server_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        video_layout.addWidget(self.server_info_label)

        # 双窗口显示（水平分割）
        video_splitter = QSplitter(Qt.Horizontal)

        # 原始视频流显示窗口
        self.label_original = QLabel("原始视频流")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b; color: white;")
        self.label_original.setMinimumSize(400, 300)
        video_splitter.addWidget(self.label_original)

        # 检测后视频流显示窗口
        self.label_detected = QLabel("检测后视频流")
        self.label_detected.setAlignment(Qt.AlignCenter)
        self.label_detected.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b; color: white;")
        self.label_detected.setMinimumSize(400, 300)
        video_splitter.addWidget(self.label_detected)

        video_layout.addWidget(video_splitter)
        splitter.addWidget(video_widget)

        # 设置分割器比例：设备列表占 1 份，视频显示占 4 份
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        # 状态栏显示服务器信息
        self.statusBar().showMessage(f"就绪 - 等待客户端连接到 ws://{local_ip}:{port}")

        # ========== 连接全局信号 ==========
        # 关键：将全局信号发射器的信号连接到本窗口的槽函数
        # 当 WebSocket 线程发射信号时，会自动调用对应的方法
        # Qt 的信号槽机制会自动处理线程切换，确保界面更新在主线程中执行
        signal_emitter.detection_result.connect(self.on_message_received)
        signal_emitter.client_connected.connect(self.on_client_connected)
        signal_emitter.client_disconnected.connect(self.on_client_disconnected)

    def on_client_connected(self, device_id, device_name, ip, port):
        """客户端连接事件处理（槽函数）.

        触发时机：当新客户端连接到服务器时
        执行线程：Qt 主线程

        功能：在设备列表中添加新设备，显示设备名称、IP 和端口

        参数：
            device_id: 设备唯一标识
            device_name: 设备名称
            ip: 客户端 IP 地址
            port: 客户端端口
        """
        # 存储设备信息
        self.devices[device_id] = {"name": device_name, "ip": ip, "port": port}

        # 在设备列表中添加新项，显示设备名称和 IP:端口
        item = QListWidgetItem(f"● {device_name}\n   {ip}:{port}")
        item.setData(Qt.UserRole, device_id)  # 存储设备 ID（用于切换设备）
        item.setForeground(QColor("green"))  # 绿色表示在线
        self.device_list.addItem(item)

        # 自动选择第一个设备
        if self.current_device_id is None:
            self.current_device_id = device_id
            self.device_list.setCurrentRow(0)

        print(f"[界面] 添加设备: {device_name} ({ip}:{port})")

    def on_client_disconnected(self, device_id):
        """客户端断开事件处理（槽函数）.

        触发时机：当客户端断开连接时
        执行线程：Qt 主线程

        功能：从设备列表中移除断开的设备

        参数：
            device_id: 设备唯一标识
        """
        # 从设备字典中移除
        if device_id in self.devices:
            device_info = self.devices.pop(device_id)
            print(f"[界面] 移除设备: {device_info['name']} ({device_info['ip']}:{device_info['port']})")

        # 从列表控件中移除
        for i in range(self.device_list.count()):
            item = self.device_list.item(i)
            if item.data(Qt.UserRole) == device_id:
                self.device_list.takeItem(i)
                break

        # 如果移除的是当前选中的设备，清空显示
        if device_id == self.current_device_id:
            self.current_device_id = None
            self.label_original.clear()
            self.label_original.setText("原始视频流")
            self.label_detected.clear()
            self.label_detected.setText("检测后视频流")

            # 如果还有其他设备，自动选择第一个
            if self.device_list.count() > 0:
                self.device_list.setCurrentRow(0)
                first_item = self.device_list.item(0)
                self.current_device_id = first_item.data(Qt.UserRole)

    def on_message_received(self, data):
        """接收检测结果（槽函数）.

        触发时机：当 WebSocket 线程发射 detection_result 信号时自动调用
        执行线程：Qt 主线程（由 Qt 信号槽机制自动切换）

        功能：
        1. 如果是当前选中的设备，更新视频显示
        2. 更新状态栏信息

        参数：
            data: 检测结果字典，包含：
                - type: "detection_result"
                - device_id: 设备唯一标识
                - device_name: 设备名称
                - original_image: 原始图像（base64）
                - detected_image: 检测后图像（base64）
                - detections: 检测结果列表

        注意：
            设备列表的更新已经由 on_client_connected 处理
            这里只需要更新视频显示即可
        """
        if data["type"] == "detection_result":
            device_id = data["device_id"]

            # ========== 更新视频显示 ==========
            # 只更新当前选中设备的视频流（避免频繁切换造成界面闪烁）
            if device_id == self.current_device_id:
                self.update_display(data)

    def update_display(self, data):
        """更新视频显示.

        功能：
        1. 解码并显示原始图像
        2. 解码并显示检测后图像
        3. 更新状态栏信息

        参数：
            data: 检测结果字典
        """
        # 解码并显示原始图像
        original_img = self.decode_base64_image(data["original_image"])
        self.display_image(self.label_original, original_img)

        # 解码并显示检测后图像
        detected_img = self.decode_base64_image(data["detected_image"])
        self.display_image(self.label_detected, detected_img)

        # 更新状态栏：显示设备名称、IP:端口 和检测到的目标数量
        num_detections = len(data["detections"])
        device_info = self.devices.get(data["device_id"], {})
        device_name = device_info.get("name", data["device_name"])
        device_ip = device_info.get("ip", "unknown")
        device_port = device_info.get("port", 0)
        self.statusBar().showMessage(
            f"设备: {device_name} ({device_ip}:{device_port}) | 检测到 {num_detections} 个目标"
        )

    def decode_base64_image(self, base64_str):
        """解码 base64 图像.

        流程：
        1. base64 字符串 -> 二进制数据
        2. 二进制数据 -> numpy 数组
        3. numpy 数组 -> OpenCV 图像矩阵

        参数：
            base64_str: base64 编码的图像字符串

        返回：
            OpenCV 图像矩阵（BGR 格式）
        """
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def display_image(self, label, cv_image):
        """在 QLabel 中显示图像.

        流程：
        1. OpenCV 图像（BGR）-> RGB 格式
        2. RGB 图像 -> QImage
        3. QImage -> QPixmap
        4. 缩放 QPixmap 以适应 QLabel 大小
        5. 设置到 QLabel

        参数：
            label: QLabel 控件
            cv_image: OpenCV 图像矩阵

        注意：
            - OpenCV 使用 BGR 格式，Qt 使用 RGB 格式，需要转换
            - 使用 KeepAspectRatio 保持图像宽高比
        """
        if cv_image is None:
            return

        # BGR -> RGB（OpenCV 和 Qt 的颜色格式不同）
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # 创建 QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 转换为 QPixmap
        pixmap = QPixmap.fromImage(qt_image)
        # 缩放以适应 QLabel 大小（保持宽高比）
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 显示图像
        label.setPixmap(scaled_pixmap)

    def on_device_selected(self, item):
        """切换设备（槽函数）.

        触发时机：用户点击设备列表中的某个设备
        功能：切换当前显示的设备视频流

        参数：
            item: QListWidgetItem 对象
        """
        # 从 item 中获取存储的 device_id
        self.current_device_id = item.data(Qt.UserRole)

    def closeEvent(self, event):
        """关闭窗口事件处理.

        触发时机：用户关闭窗口时
        功能：清理资源（当前无需特殊清理，WebSocket 线程是 daemon 线程会自动退出）

        参数：
            event: 关闭事件对象
        """
        event.accept()


# ============================================================================
# 主程序入口
# ============================================================================


def main():
    """主函数.

    执行流程：
    1. 启动 WebSocket 服务器（后台线程）
    2. 启动 PyQt5 监控界面（主线程）

    线程架构：
    ┌─────────────────────────────────────────────────────────┐
    │  主线程                                                  │
    │  ├─ PyQt5 事件循环                                      │
    │  └─ MonitorWindow 界面更新                              │
    └─────────────────────────────────────────────────────────┘
                        ↑ Qt 信号（线程安全）
    ┌─────────────────────────────────────────────────────────┐
    │  后台线程（daemon）                                      │
    │  ├─ asyncio 事件循环                                    │
    │  ├─ WebSocket 服务器                                    │
    │  ├─ 接收客户端消息                                      │
    │  ├─ YOLO 检测                                           │
    │  └─ 发射 Qt 信号                                        │
    └─────────────────────────────────────────────────────────┘

    为什么使用 daemon 线程：
    - daemon=True 表示守护线程
    - 当主线程（PyQt 应用）退出时，守护线程会自动终止
    - 无需手动管理 WebSocket 服务器的关闭
    """
    # ========== 步骤 1: 启动 WebSocket 服务器 ==========
    # 在后台线程中运行 WebSocket 服务器
    server_thread = threading.Thread(target=run_websocket_server, daemon=True)
    server_thread.start()

    # ========== 步骤 2: 启动 PyQt 应用 ==========
    # 创建 Qt 应用实例
    app = QApplication(sys.argv)
    # 创建监控窗口
    window = MonitorWindow()
    # 显示窗口
    window.show()
    # 进入 Qt 事件循环（阻塞，直到窗口关闭）
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
