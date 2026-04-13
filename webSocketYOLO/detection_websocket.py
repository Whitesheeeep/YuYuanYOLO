"""
YuYuan YOLO 检测服务（WebSocket 服务器）
=========================================

功能概述：
1. 作为 WebSocket 服务器，接收来自 Unity 客户端的图像数据
2. 使用 YOLO 模型进行实时目标检测（绘框、发送结果给 Unity）
3. 通过 Qt 信号槽机制向外部（监控界面）传递检测结果
4. 支持多设备同时连接

导览 Agent 架构（YuYuanGuidanceAgent）：
- YOLO 模型实例在服务启动时加载，通过构造函数注入 Agent
- Agent 内部持有三个工具：rag_search / yolo_detect / image_understand
- 图像存入 AgentState.current_image_base64，LLM 初始轮仅见文字；
  工具按需从 state 读取图像，避免每轮重复传输大体积 base64
- 会话记忆由 InMemorySaver checkpointer 管理，按 session_id 隔离

数据流向：
Unity 客户端 --WebSocket--> 服务器 --YOLO检测--> 信号发射器 --Qt信号--> 监控界面
                                    └--> 回发检测结果 --> Unity 客户端
Unity 客户端 --guidance_request--> Agent(RAG/YOLO/Vision tools) --> 回发导览答案
"""
import sys
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import hashlib
import struct
from pathlib import Path
from ultralytics import YOLO
import threading
import socket
import ButtonConfig

# 允许从项目根目录导入 LangChain 模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LangChain.guidance_agent import YuYuanGuidanceAgent

# ============================================================================
# 全局配置和初始化
# ============================================================================

def get_local_ip():
    """获取本机局域网 IP 地址。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

# 加载 YOLO 模型
model = YOLO(r'D:\master\Yuyuan2\YuyuanYOLO\runs\detect\runs\train\yuyuan_exp\weights\best.pt')

# WebSocket 服务器配置
ip = "0.0.0.0"
port = 5000
local_ip = get_local_ip()

# Binary protocol magic: "YY" = 0x5959
BINARY_MAGIC = 0x5959
BINARY_VERSION = 1

# Binary MessageType
class MsgType:
    DETECTION = 0
    GUIDANCE = 1
    DETECTION_RESULT = 2
    GUIDANCE_RESULT = 3

# 历史保留策略
AGENT_HISTORY_LIMIT = 10

# ============================================================================
# 按钮配置（修改 name 即可自定义按钮名称）
# ============================================================================

# 存储所有连接的 WebSocket 客户端
clients = {}

# 存储所有连接的控制客户端（key = connection_id）
control_clients = {}

# asyncio event loop（供跨线程调用，由 run_websocket_server 设置）
_ws_loop = None

_guidance_agent = None


def get_guidance_agent():
    """延迟初始化导览 Agent。"""
    global _guidance_agent
    if _guidance_agent is not None:
        return _guidance_agent
    try:
        _guidance_agent = YuYuanGuidanceAgent(
            yolo_model=model,
            agent_history_limit=AGENT_HISTORY_LIMIT,
        )
        print(f"[Agent] 导览 Agent 初始化成功，history_limit={AGENT_HISTORY_LIMIT}")
    except Exception as e:
        _guidance_agent = None
        print(f"[Agent Error] 初始化失败: {e}")
    return _guidance_agent


# 用户可配置的类别颜色（OpenCV BGR 格式）
USER_CLASS_COLORS = {}

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
    """类别颜色映射，支持用户覆盖并提供稳定的默认色。"""
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
    """客户端会话类，存储单个客户端的连接信息和状态。"""
    def __init__(self, connection_id, device_id, device_name, websocket, remote_address):
        self.connection_id = connection_id
        self.device_id = device_id
        self.device_name = device_name
        self.websocket = websocket
        self.remote_address = remote_address
        self.connected_at = asyncio.get_event_loop().time()
        self.last_original_image = None
        self.guidance_tasks = set()

    @property
    def ip(self):
        return self.remote_address[0] if self.remote_address else "unknown"

    @property
    def port(self):
        return self.remote_address[1] if self.remote_address else 0

    def __repr__(self):
        return f"ClientSession(conn_id={self.connection_id}, device_id={self.device_id}, device_name={self.device_name}, ip={self.ip}, port={self.port})"


# ============================================================================
# 线程间通信机制
# ============================================================================

try:
    from PyQt5.QtCore import QObject, pyqtSignal
    _HAS_PYQT = True
except ImportError:
    _HAS_PYQT = False
    # 纯 asyncio 模式下的信号替代（不触发 Qt 信号）
    class _DummySignal:
        def emit(self, *args): pass
    class QObject:
        pass
    class pyqtSignal:
        def __init__(self, *types): pass
        def connect(self, *args): pass


class SignalEmitter(QObject):
    """全局信号发射器，实现从 WebSocket 线程向 Qt 主线程传递数据。"""
    detection_result = pyqtSignal(dict, str)
    guidance_result = pyqtSignal(dict, str)
    client_connected = pyqtSignal(str, str, str, str, int)
    client_disconnected = pyqtSignal(str)
    history_cleared = pyqtSignal(str)  # connection_id


signal_emitter = SignalEmitter()

# ============================================================================
# 辅助函数
# ============================================================================


def apply_nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制（NMS）。"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def make_temp_device_id(connection_id: str) -> str:
    safe = connection_id.replace(":", "_")
    return f"temp_device_{safe}"


def make_session_id(connection_id: str, device_id: str) -> str:
    return f"{connection_id}_{device_id}"


async def generate_guidance_answer(query: str, session_id: str, image_base64: str) -> str:
    agent = get_guidance_agent()
    if agent is None:
        return "导览助手暂时不可用，请稍后再试。"
    return await agent.generate_guidance_with_image(
        user_query=query,
        session_id=session_id,
        image_base64=image_base64,
    )


async def process_guidance_request(connection_id: str, device_id: str, device_name: str, query: str, image_base64: str):
    """后台处理导览请求。"""
    try:
        session_id = make_session_id(connection_id, device_id)
        answer = await generate_guidance_answer(query, session_id, image_base64)
    except asyncio.CancelledError:
        return
    except Exception as e:
        answer = f"导览请求处理失败: {e}"

    response = {
        'type': 'guidance_result',
        'device_id': device_id,
        'device_name': device_name,
        'query': query,
        'answer': answer,
    }

    session = clients.get(connection_id)
    if session is not None:
        print(f"[Guidance] session={make_session_id(connection_id, device_id)} 答复已发送")

    signal_emitter.guidance_result.emit(response, connection_id)
    await send_to_client(connection_id, json.dumps(response, ensure_ascii=False))


# ============================================================================
# Binary Protocol 辅助函数
# ============================================================================

def pack_binary_frame(msg_type, metadata: dict, image_bytes: bytes):
    """
    打包 binary frame: [Header(16)][Metadata JSON][Image]
    Header:
      - Magic (4 bytes, little-endian uint32) = 0x5959
      - Version (1 byte) = 1
      - MessageType (1 byte)
      - Reserved (2 bytes) = 0
      - MetadataLength (4 bytes, big-endian)
      - ImageLength (4 bytes, big-endian)
    """
    meta_json = json.dumps(metadata, ensure_ascii=False)
    meta_bytes = meta_json.encode('utf-8')
    meta_len = len(meta_bytes)
    image_len = len(image_bytes)

    header = struct.pack('<IBBH', BINARY_MAGIC, BINARY_VERSION, msg_type, 0)
    header += struct.pack('>II', meta_len, image_len)  # big-endian
    return header + meta_bytes + image_bytes


def parse_binary_request(data: bytes):
    """
    解析 binary request frame。
    Returns: (msg_type, device_id, device_name, metadata, image_bytes)
             or (None, None, None, None, None) on error.
    """
    if len(data) < 16:
        return None, None, None, None, None

    magic, version, msg_type = struct.unpack_from('<IBB', data, 0)
    if magic != BINARY_MAGIC:
        return None, None, None, None, None

    meta_len, image_len = struct.unpack_from('>II', data, 8)
    header_end = 16
    if header_end + meta_len + image_len > len(data):
        return None, None, None, None, None

    meta_json = data[header_end:header_end + meta_len].decode('utf-8')
    image_bytes = data[header_end + meta_len:header_end + meta_len + image_len]

    meta = json.loads(meta_json)
    device_id = meta.get('device_id', '')
    device_name = meta.get('device_name', '')
    return msg_type, device_id, device_name, meta, image_bytes


# ============================================================================
# WebSocket 服务器核心逻辑
# ============================================================================


async def broadcast_client_list_to_controllers():
    """向所有控制客户端推送当前检测客户端列表。"""
    if not control_clients:
        return
    payload = json.dumps({
        'type': 'detection_client_list',
        'clients': [
            {
                'connection_id': s.connection_id,
                'device_id': s.device_id,
                'device_name': s.device_name,
            }
            for s in clients.values()
        ]
    }, ensure_ascii=False)
    await asyncio.gather(
        *[ws.send(payload) for ws in control_clients.values()],
        return_exceptions=True
    )


async def handle_client(websocket):
    """处理单个客户端连接。"""
    remote_address = websocket.remote_address
    connection_id = f"{remote_address[0]}:{remote_address[1]}"
    print(f"新连接来自: {remote_address[0]}:{remote_address[1]}, connection_id={connection_id}")

    current_connection_id = None

    try:
        def ensure_session(device_id: str, device_name: str):
            """返回 (ClientSession, is_new)"""
            nonlocal current_connection_id
            if connection_id not in clients:
                session_obj = ClientSession(connection_id, device_id, device_name, websocket, remote_address)
                clients[connection_id] = session_obj
                current_connection_id = connection_id
                print(f"[注册] 新设备: {session_obj}")
                print(f"当前连接设备数: {len(clients)}")
                signal_emitter.client_connected.emit(connection_id, device_id, device_name, session_obj.ip, session_obj.port)
                return session_obj, True

            session_obj = clients[connection_id]
            session_obj.websocket = websocket
            session_obj.remote_address = remote_address
            session_obj.device_id = device_id
            session_obj.device_name = device_name
            current_connection_id = connection_id
            return session_obj, False

        async for message in websocket:
            # ============================================================
            # Binary 帧路径（新协议）
            # ============================================================
            if isinstance(message, bytes):
                msg_type, device_id, device_name, metadata, image_bytes = parse_binary_request(message)
                if device_id is None:
                    print(f"[警告] 二进制帧格式错误，无法解析")
                    continue

                device_id = device_id or make_temp_device_id(connection_id)
                device_name = device_name or device_id
                session, is_new = ensure_session(device_id, device_name)
                if is_new:
                    await broadcast_client_list_to_controllers()

                # ---- Guidance 二进制请求 ----
                if msg_type == MsgType.GUIDANCE:
                    query = (metadata.get('query') or '').strip()
                    if not query:
                        response = {
                            'type': 'guidance_result',
                            'device_id': session.device_id,
                            'device_name': session.device_name,
                            'query': '',
                            'answer': '请先输入导览问题。',
                        }
                        signal_emitter.guidance_result.emit(response, connection_id)
                        await send_to_client(connection_id, json.dumps(response, ensure_ascii=False))
                        continue

                    request_image_base64 = base64.b64encode(image_bytes).decode()

                    task = asyncio.create_task(
                        process_guidance_request(
                            connection_id=connection_id,
                            device_id=session.device_id,
                            device_name=session.device_name,
                            query=query,
                            image_base64=request_image_base64,
                        )
                    )
                    session.guidance_tasks.add(task)
                    task.add_done_callback(lambda t, s=session: s.guidance_tasks.discard(t))
                    continue

                # ---- Detection 二进制请求 ----
                img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[错误] 无法解码图像")
                    continue

            # ============================================================
            # Text/JSON 帧路径（旧协议兼容）
            # ============================================================
            else:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[警告] 无法解析 JSON 消息")
                    continue

                message_type = data.get('type')
                incoming_device_id = data.get('device_id') or make_temp_device_id(connection_id)
                incoming_device_name = data.get('device_name') or incoming_device_id

                if message_type == 'detect':
                    device_id = incoming_device_id
                    device_name = incoming_device_name
                    session, is_new = ensure_session(device_id, device_name)
                    if is_new:
                        await broadcast_client_list_to_controllers()

                    try:
                        image_data = base64.b64decode(data.get('image', ''))
                        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            print(f"[错误] 无法解码图像")
                            continue
                    except Exception as e:
                        print(f"[错误] 图像解码失败: {e}")
                        continue

                elif message_type == 'guidance_request':
                    device_id = incoming_device_id
                    device_name = incoming_device_name
                    session, is_new = ensure_session(device_id, device_name)
                    if is_new:
                        await broadcast_client_list_to_controllers()

                    query = (data.get('query') or data.get('message') or data.get('user_query') or '').strip()
                    if not query:
                        response = {
                            'type': 'guidance_result',
                            'device_id': session.device_id,
                            'device_name': session.device_name,
                            'query': '',
                            'answer': '请先输入导览问题。',
                        }
                        signal_emitter.guidance_result.emit(response, connection_id)
                        await send_to_client(connection_id, json.dumps(response, ensure_ascii=False))
                        continue

                    request_image_base64 = (
                        data.get('image_base64') or data.get('image') or data.get('original_image') or ''
                    ).strip()
                    if request_image_base64.startswith('data:') and ',' in request_image_base64:
                        request_image_base64 = request_image_base64.split(',', 1)[1].strip()

                    if not request_image_base64:
                        response = {
                            'type': 'guidance_result',
                            'device_id': session.device_id,
                            'device_name': session.device_name,
                            'query': query,
                            'answer': '请在 guidance_request 中传入图片 base64（image_base64 或 image 字段）。',
                        }
                        signal_emitter.guidance_result.emit(response, connection_id)
                        await send_to_client(connection_id, json.dumps(response, ensure_ascii=False))
                        continue

                    task = asyncio.create_task(
                        process_guidance_request(
                            connection_id=connection_id,
                            device_id=session.device_id,
                            device_name=session.device_name,
                            query=query,
                            image_base64=request_image_base64,
                        )
                    )
                    session.guidance_tasks.add(task)
                    task.add_done_callback(lambda t, s=session: s.guidance_tasks.discard(t))
                    continue

                elif message_type == 'register_control':
                    client_id = data.get('client_id') or connection_id
                    control_clients[connection_id] = websocket
                    current_connection_id = connection_id  # 记录用于断开时清理
                    print(f"[控制端] 注册: client_id={client_id}, connection_id={connection_id}")
                    await broadcast_client_list_to_controllers()
                    continue

                elif message_type == 'send_command':
                    target_id = data.get('target_connection_id', '')
                    button_id = data.get('button_id', 0)
                    button_name = data.get('button_name', '')
                    cmd = json.dumps({
                        'type': 'command',
                        'button_id': button_id,
                        'button_name': button_name,
                    }, ensure_ascii=False)
                    if target_id in clients:
                        await send_to_client(target_id, cmd)
                        print(f"[命令] 转发 button_id={button_id}({button_name}) → {target_id}")
                    else:
                        print(f"[命令] 目标 {target_id} 不存在，已忽略")
                    continue

                elif message_type == "query":
                    target_id = data.get('target_connection_id', '')
                    query = data.get('query', '')
                    cmd = json.dumps({
                        'type': 'query',
                        'query': query
                    }, ensure_ascii=False)
                    if target_id in clients:
                        await send_to_client(target_id, cmd)
                        print(f"[命令] 转发 query={query} → {target_id}")
                    else:
                        print(f"[命令] 目标 {target_id} 不存在，已忽略")
                    continue

                elif message_type == 'clear_history':
                    target_id = data.get('target_connection_id', '') or connection_id
                    if _guidance_agent is not None:
                        session_id = target_id
                        _guidance_agent.clear_session_history(session_id)
                        print(f"[清除历史] 已清除会话: {session_id}")
                    else:
                        print(f"[清除历史] Agent 未初始化，跳过")
                    signal_emitter.history_cleared.emit(target_id)
                    continue

                else:
                    print(f"[警告] 未知消息类型: {message_type}")
                    continue

            # ============================================================
            # YOLO 检测 + 响应（Binary/Text 两条路径均执行此处）
            # ============================================================
            # YOLO 检测
            results = model(img, verbose=False)

            all_boxes = []
            all_scores = []
            all_classes = []
            all_class_names = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)
                    all_classes.append(cls)
                    all_class_names.append(class_name)

            # NMS 过滤
            keep_indices = apply_nms(all_boxes, all_scores, iou_threshold=0.5) if all_boxes else []

            # 绘制检测框
            detected_img = img.copy()
            detections = []

            for idx in keep_indices:
                x1, y1, x2, y2 = all_boxes[idx]
                conf = all_scores[idx]
                class_name = all_class_names[idx]
                class_index = all_classes[idx]
                box_color = class_color_map.get(class_name, class_index)
                cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.putText(detected_img, f'{class_name} {conf:.2f}',
                          (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

            # 根据原始帧类型选择响应方式
            is_binary_request = isinstance(message, bytes)

            # 编码图像（original 仅供 monitor/guidance 使用，不发送给客户端）
            _, original_jpg = cv2.imencode('.jpg', img)
            _, detected_jpg = cv2.imencode('.jpg', detected_img)
            original_b64 = base64.b64encode(original_jpg).decode()
            detected_b64 = base64.b64encode(detected_jpg).decode()

            # 保存原图供 guidance 使用
            session.last_original_image = original_b64

            # 通知 monitor UI（保留 original_image 供本地显示）
            signal_emitter.detection_result.emit(
                {'type': 'detection_result', 'device_id': device_id, 'device_name': device_name,
                 'original_image': original_b64, 'detected_image': detected_b64,
                 'detections': detections},
                connection_id
            )

            if is_binary_request:
                # ---- Binary 响应：header + metadata + detected_image（仅检测图）----
                meta = {
                    'type': 'detection_result',
                    'device_id': device_id,
                    'device_name': device_name,
                    'detections': detections,
                }
                response_frame = pack_binary_frame(MsgType.DETECTION_RESULT, meta, detected_jpg.tobytes())
                await send_to_client(connection_id, response_frame)
            else:
                # ---- Text 响应（旧 JSON 协议）----
                response = {
                    'type': 'detection_result',
                    'device_id': device_id,
                    'device_name': device_name,
                    'detected_image': detected_b64,
                    'detections': detections
                }
                await send_to_client(connection_id, json.dumps(response))

            if len(detections) > 0:
                print(f"[检测] 设备 {device_name} ({session.ip}:{session.port}) - 检测到 {len(detections)} 个目标 (NMS 过滤后)")

    except websockets.exceptions.ConnectionClosed:
        print(f"[断开] 客户端断开连接: {remote_address[0]}:{remote_address[1]}")
    except Exception as e:
        print(f"[错误] 处理客户端消息时出错: {e}")
    finally:
        if current_connection_id and current_connection_id in control_clients:
            control_clients.pop(current_connection_id)
            print(f"[移除] 控制客户端: {current_connection_id}")
        elif current_connection_id and current_connection_id in clients:
            removed_session = clients.pop(current_connection_id)
            print(f"[移除] 设备: {removed_session}")
            print(f"当前连接设备数: {len(clients)}")
            for task in list(removed_session.guidance_tasks):
                task.cancel()
            removed_session.guidance_tasks.clear()
            signal_emitter.client_disconnected.emit(current_connection_id)
            await broadcast_client_list_to_controllers()


async def send_to_client(connection_id, message):
    """发送消息给指定的客户端。"""
    if connection_id in clients:
        session = clients[connection_id]
        try:
            await session.websocket.send(message)
        except Exception as e:
            print(f"[错误] 发送消息到连接 {connection_id} (设备: {session.device_id}) 失败: {e}")
    else:
        print(f"[警告] 连接 {connection_id} 不在连接列表中")


async def broadcast(message):
    """广播消息到所有连接的客户端。"""
    if clients:
        await asyncio.gather(
            *[session.websocket.send(message) for session in clients.values()],
            return_exceptions=True
        )


async def start_websocket_server():
    """启动 WebSocket 服务器。"""
    print("=" * 60)
    print("启动 WebSocket 服务器（独立模式）")
    print("=" * 60)
    print(f"监听地址: ws://{ip}:{port} (所有网络接口)")
    print(f"局域网连接地址: ws://{local_ip}:{port}")
    print(f"本地连接地址: ws://127.0.0.1:{port}")
    print("=" * 60)

    # 预热 agent
    get_guidance_agent()

    async with websockets.serve(handle_client, ip, port):
        await asyncio.Future()


def run_websocket_server():
    """在单独线程中运行 WebSocket 服务器。"""
    global _ws_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _ws_loop = loop
    loop.run_until_complete(start_websocket_server())


# ============================================================================
# 主程序入口（独立运行，无 PyQt 界面）
# ============================================================================


def main():
    """独立启动 WebSocket 检测服务（无 PyQt 界面）。"""
    print("=" * 60)
    print("YuYuan YOLO 检测服务（独立模式）")
    print("=" * 60)
    print("WebSocket 服务器即将启动...")
    print(f"监听地址: ws://{local_ip}:{port}")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)

    # 预热 Guidance Agent
    get_guidance_agent()

    try:
        run_websocket_server()
    except KeyboardInterrupt:
        print("\n服务器已停止。")


if __name__ == '__main__':
    main()
