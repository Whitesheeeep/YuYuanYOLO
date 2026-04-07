"""
YuYuan YOLO 检测监控系统（PyQt5 可视化界面）
=============================================

功能概述：
1. 显示所有连接的设备列表
2. 显示选中设备的原始视频流和检测后视频流
3. 实时更新检测结果和导览对话

数据流向：
detection_websocket.py (WS 服务器) --Qt信号--> MonitorWindow (界面显示)

使用方法：
- 运行本文件：启动 WS 后台服务 + PyQt5 界面（完整监控模式）
- detection_websocket.py 可独立运行，作为无界面的检测服务
"""
import sys
import threading
import json
import base64
import cv2
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QSplitter, QSizePolicy,
                             QTextEdit, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont

# 从 WS 模块导入信号发射器和服务器启动函数
from detection_websocket import signal_emitter, run_websocket_server, local_ip, port

# 历史保留策略
MONITOR_HISTORY_LIMIT_DEFAULT = 20


# ============================================================================
# PyQt5 监控界面
# ============================================================================


class MonitorWindow(QMainWindow):
    """
    监控窗口主类

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
        self.setWindowTitle('YuYuan 检测监控系统')
        self.setGeometry(100, 100, 1400, 900)

        # 存储设备信息：{connection_id: {'name': device_name, 'device_id': device_id, 'ip': ip, 'port': port}}
        self.devices = {}
        # 当前选中的 connection_id
        self.current_connection_id = None
        # monitor 展示历史
        self.monitor_history = {}
        self.monitor_history_limit = MONITOR_HISTORY_LIMIT_DEFAULT

        # 初始化界面
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面。"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter()

        # ========== 左侧：设备列表 ==========
        self.device_list = QListWidget()
        self.device_list.itemClicked.connect(self.on_device_selected)
        splitter.addWidget(self.device_list)

        # ========== 右侧：视频显示区域 ==========
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # 服务器信息标签
        self.server_info_label = QLabel()
        self.server_info_label.setAlignment(Qt.AlignCenter)
        self.server_info_label.setStyleSheet(
            'background-color: #1e1e1e; color: #00ff00; '
            'padding: 10px; border: 2px solid #00ff00; '
            'border-radius: 5px; font-weight: bold;'
        )
        server_info_text = (
            f"WebSocket 服务器运行中\n"
            f"局域网地址: ws://{local_ip}:{port}\n"
            f"本地地址: ws://127.0.0.1:{port}"
        )
        self.server_info_label.setText(server_info_text)
        self.server_info_label.setFont(QFont("Consolas", 10))
        self.server_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        video_layout.addWidget(self.server_info_label)

        # 对话历史保留条数控制
        history_control_layout = QHBoxLayout()
        history_label = QLabel('历史保留轮数:')
        self.history_limit_spin = QSpinBox()
        self.history_limit_spin.setRange(1, 200)
        self.history_limit_spin.setValue(self.monitor_history_limit)
        self.history_limit_spin.valueChanged.connect(self.on_history_limit_changed)
        history_control_layout.addWidget(history_label)
        history_control_layout.addWidget(self.history_limit_spin)
        history_control_layout.addStretch(1)
        video_layout.addLayout(history_control_layout)

        # 双窗口显示（水平分割）
        video_splitter = QSplitter(Qt.Horizontal)

        # 原始视频流显示窗口
        self.label_original = QLabel('原始视频流')
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet('border: 1px solid gray; background-color: #2b2b2b; color: white;')
        self.label_original.setMinimumSize(400, 300)
        video_splitter.addWidget(self.label_original)

        # 检测后视频流显示窗口
        self.label_detected = QLabel('检测后视频流')
        self.label_detected.setAlignment(Qt.AlignCenter)
        self.label_detected.setStyleSheet('border: 1px solid gray; background-color: #2b2b2b; color: white;')
        self.label_detected.setMinimumSize(400, 300)
        video_splitter.addWidget(self.label_detected)

        video_layout.addWidget(video_splitter)

        # 当前设备对话历史
        self.chat_history_view = QTextEdit()
        self.chat_history_view.setReadOnly(True)
        self.chat_history_view.setPlaceholderText('当前设备对话历史将显示在这里...')
        video_layout.addWidget(self.chat_history_view)

        splitter.addWidget(video_widget)

        # 设置分割器比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        # 状态栏显示服务器信息
        self.statusBar().showMessage(f'就绪 - 等待客户端连接到 ws://{local_ip}:{port}')

        # ========== 连接全局信号 ==========
        signal_emitter.detection_result.connect(self.on_message_received)
        signal_emitter.guidance_result.connect(self.on_guidance_received)
        signal_emitter.client_connected.connect(self.on_client_connected)
        signal_emitter.client_disconnected.connect(self.on_client_disconnected)

    def on_client_connected(self, connection_id, device_id, device_name, ip, port):
        """客户端连接事件处理。"""
        self.devices[connection_id] = {
            'name': device_name,
            'device_id': device_id,
            'ip': ip,
            'port': port
        }
        if connection_id not in self.monitor_history:
            self.monitor_history[connection_id] = []

        display_text = f"* {device_name} [{device_id}]\n   {connection_id}"
        item = QListWidgetItem(display_text)
        item.setData(Qt.UserRole, connection_id)
        item.setForeground(QColor('green'))
        self.device_list.addItem(item)

        if self.current_connection_id is None:
            self.current_connection_id = connection_id
            self.device_list.setCurrentRow(0)

        print(f"[界面] 添加设备: {device_name} [{device_id}] @ {connection_id}")

    def on_client_disconnected(self, connection_id):
        """客户端断开事件处理。"""
        if connection_id in self.devices:
            device_info = self.devices.pop(connection_id)
            print(f"[界面] 移除设备: {device_info['name']} [{device_info['device_id']}] @ {connection_id}")
        self.monitor_history.pop(connection_id, None)

        for i in range(self.device_list.count()):
            item = self.device_list.item(i)
            if item.data(Qt.UserRole) == connection_id:
                self.device_list.takeItem(i)
                break

        if connection_id == self.current_connection_id:
            self.current_connection_id = None
            self.label_original.clear()
            self.label_original.setText('原始视频流')
            self.label_detected.clear()
            self.label_detected.setText('检测后视频流')
            self.chat_history_view.clear()

            if self.device_list.count() > 0:
                self.device_list.setCurrentRow(0)
                first_item = self.device_list.item(0)
                self.current_connection_id = first_item.data(Qt.UserRole)

    def on_message_received(self, data, connection_id):
        """接收检测结果。"""
        if data['type'] == 'detection_result':
            if connection_id == self.current_connection_id:
                self.update_display(data, connection_id)

    def on_guidance_received(self, data, connection_id):
        """接收导览结果。"""
        if data.get('type') != 'guidance_result':
            return
        query = data.get('query', '').strip()
        answer = data.get('answer', '').strip()
        if query:
            self.append_and_trim_history(connection_id, f"[User] {query}")
        if answer:
            self.append_and_trim_history(connection_id, f"[AI] {answer}")
        if connection_id == self.current_connection_id:
            self.refresh_history_display()

    def append_and_trim_history(self, connection_id, line):
        history = self.monitor_history.setdefault(connection_id, [])
        history.append(line)
        max_lines = max(2, self.monitor_history_limit * 2)
        if len(history) > max_lines:
            del history[:-max_lines]

    def refresh_history_display(self):
        if not self.current_connection_id:
            self.chat_history_view.clear()
            return
        lines = self.monitor_history.get(self.current_connection_id, [])
        self.chat_history_view.setPlainText("\n".join(lines))

    def on_history_limit_changed(self, value):
        self.monitor_history_limit = max(1, int(value))
        max_lines = max(2, self.monitor_history_limit * 2)
        for connection_id, lines in self.monitor_history.items():
            if len(lines) > max_lines:
                self.monitor_history[connection_id] = lines[-max_lines:]
        self.refresh_history_display()

    def update_display(self, data, connection_id):
        """更新视频显示。"""
        original_img = self.decode_base64_image(data['original_image'])
        self.display_image(self.label_original, original_img)

        detected_img = self.decode_base64_image(data['detected_image'])
        self.display_image(self.label_detected, detected_img)

        num_detections = len(data['detections'])
        device_info = self.devices.get(connection_id, {})
        device_name = device_info.get('name', data['device_name'])
        device_ip = device_info.get('ip', 'unknown')
        device_port = device_info.get('port', 0)
        self.statusBar().showMessage(
            f"设备: {device_name} ({device_ip}:{device_port}) | 检测到 {num_detections} 个目标"
        )

    def decode_base64_image(self, base64_str):
        """解码 base64 图像。"""
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def display_image(self, label, cv_image):
        """在 QLabel 中显示图像。"""
        if cv_image is None:
            return
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def on_device_selected(self, item):
        """切换设备。"""
        self.current_connection_id = item.data(Qt.UserRole)
        self.refresh_history_display()

    def closeEvent(self, event):
        """关闭窗口事件处理。"""
        event.accept()


# ============================================================================
# 主程序入口
# ============================================================================


def main():
    """
    主函数

    执行流程：
    1. 启动 WebSocket 服务器（后台线程）
    2. 启动 PyQt5 监控界面（主线程）
    """
    # ========== 步骤 1: 启动 WebSocket 服务器 ==========
    server_thread = threading.Thread(target=run_websocket_server, daemon=True)
    server_thread.start()

    # ========== 步骤 2: 启动 PyQt 应用 ==========
    app = QApplication(sys.argv)
    window = MonitorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
