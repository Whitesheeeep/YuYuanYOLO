import asyncio
import base64
import json
import sys

import cv2
import numpy as np
import websockets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class WebSocketThread(QThread):
    """WebSocket 接收线程."""

    message_received = pyqtSignal(dict)

    def __init__(self, server_url):
        super().__init__()
        self.server_url = server_url
        self.running = True

    def run(self):
        asyncio.run(self.connect())

    async def connect(self):
        try:
            async with websockets.connect(self.server_url) as websocket:
                print(f"已连接到服务器: {self.server_url}")

                while self.running:
                    message = await websocket.recv()
                    data = json.loads(message)
                    self.message_received.emit(data)

        except Exception as e:
            print(f"连接错误: {e}")

    def stop(self):
        self.running = False


class MonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YuYuan 检测监控系统")
        self.setGeometry(100, 100, 1400, 900)

        self.devices = {}  # device_id -> device_info
        self.current_device_id = None

        self.setup_ui()
        self.start_websocket()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter()

        # 左侧：设备列表
        self.device_list = QListWidget()
        self.device_list.itemClicked.connect(self.on_device_selected)
        splitter.addWidget(self.device_list)

        # 右侧：视频显示
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # 双窗口显示
        video_splitter = QSplitter(Qt.Horizontal)

        self.label_original = QLabel("原始视频流")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.label_original.setMinimumSize(400, 300)
        video_splitter.addWidget(self.label_original)

        self.label_detected = QLabel("检测后视频流")
        self.label_detected.setAlignment(Qt.AlignCenter)
        self.label_detected.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.label_detected.setMinimumSize(400, 300)
        video_splitter.addWidget(self.label_detected)

        video_layout.addWidget(video_splitter)
        splitter.addWidget(video_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)

        self.statusBar().showMessage("就绪")

    def start_websocket(self):
        """启动 WebSocket 连接."""
        self.ws_thread = WebSocketThread("ws://localhost:5000")
        self.ws_thread.message_received.connect(self.on_message_received)
        self.ws_thread.start()

    def on_message_received(self, data):
        """接收到检测结果."""
        if data["type"] == "detection_result":
            device_id = data["device_id"]
            device_name = data["device_name"]

            # 更新设备列表
            if device_id not in self.devices:
                self.devices[device_id] = device_name
                item = QListWidgetItem(f"● {device_name}")
                item.setData(Qt.UserRole, device_id)
                item.setForeground(QColor("green"))
                self.device_list.addItem(item)

                # 自动选择第一个设备
                if self.current_device_id is None:
                    self.current_device_id = device_id
                    self.device_list.setCurrentRow(0)

            # 如果是当前选中的设备，更新显示
            if device_id == self.current_device_id:
                self.update_display(data)

    def update_display(self, data):
        """更新视频显示."""
        # 解码原始图像
        original_img = self.decode_base64_image(data["original_image"])
        self.display_image(self.label_original, original_img)

        # 解码检测后图像
        detected_img = self.decode_base64_image(data["detected_image"])
        self.display_image(self.label_detected, detected_img)

        # 更新状态栏
        num_detections = len(data["detections"])
        self.statusBar().showMessage(f"设备: {data['device_name']} | 检测到 {num_detections} 个目标")

    def decode_base64_image(self, base64_str):
        """解码 base64 图像."""
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def display_image(self, label, cv_image):
        """显示图像."""
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
        """切换设备."""
        self.current_device_id = item.data(Qt.UserRole)

    def closeEvent(self, event):
        """关闭窗口."""
        self.ws_thread.stop()
        self.ws_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MonitorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
