"""
YuYuan 控制客户端（PyQt5）
===========================
连接 WebSocket 服务端，注册为控制客户端，选择目标检测客户端并发送命令按钮。

使用方法：
    python control_client.py
"""

import sys
import asyncio
import json
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QGridLayout, QStatusBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
import websockets

# ============================================================================
# 按钮配置（与服务端 BUTTON_CONFIGS 保持一致，修改 name 自定义名称）
# ============================================================================

BUTTON_CONFIGS = [
    {"id": 1,  "name": "按钮1"},
    {"id": 2,  "name": "按钮2"},
    {"id": 3,  "name": "按钮3"},
    {"id": 4,  "name": "按钮4"},
    {"id": 5,  "name": "按钮5"},
    {"id": 6,  "name": "按钮6"},
    {"id": 7,  "name": "按钮7"},
    {"id": 8,  "name": "按钮8"},
    {"id": 9,  "name": "按钮9"},
    {"id": 10, "name": "按钮10"},
]

CLIENT_ID = "control_client_python"

# ============================================================================
# 信号桥（asyncio → Qt 主线程）
# ============================================================================


class _Bridge(QObject):
    client_list_updated = pyqtSignal(list)   # list of {connection_id, device_id, device_name}
    status_changed = pyqtSignal(str)


bridge = _Bridge()

# ============================================================================
# asyncio WebSocket 管理（运行在子线程）
# ============================================================================

_ws = None
_ws_loop = None


def _start_ws_thread(url: str):
    global _ws_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _ws_loop = loop
    loop.run_until_complete(_connect(url))


async def _connect(url: str):
    global _ws
    bridge.status_changed.emit(f"正在连接 {url} ...")
    try:
        async with websockets.connect(url) as ws:
            _ws = ws
            bridge.status_changed.emit(f"已连接: {url}")

            # 注册为控制客户端
            await ws.send(json.dumps({"type": "register_control", "client_id": CLIENT_ID}))

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "detection_client_list":
                    bridge.client_list_updated.emit(msg.get("clients", []))
    except Exception as e:
        bridge.status_changed.emit(f"连接断开: {e}")
    finally:
        _ws = None


def send_command(target_connection_id: str, button_id: int, button_name: str):
    """从 Qt 线程安全地发送命令到 asyncio 循环。"""
    if _ws is None or _ws_loop is None:
        bridge.status_changed.emit("未连接，无法发送命令")
        return
    payload = json.dumps({
        "type": "send_command",
        "target_connection_id": target_connection_id,
        "button_id": button_id,
        "button_name": button_name,
    }, ensure_ascii=False)
    asyncio.run_coroutine_threadsafe(_ws.send(payload), _ws_loop)

# ============================================================================
# 主窗口
# ============================================================================


class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YuYuan 控制客户端")
        self.setMinimumWidth(560)
        self._detection_clients = []   # list of dict
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # ---- 连接区 ----
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("IP:"))
        self.ip_edit = QLineEdit("192.168.1.100")
        self.ip_edit.setFixedWidth(150)
        conn_layout.addWidget(self.ip_edit)
        conn_layout.addWidget(QLabel("Port:"))
        self.port_edit = QLineEdit("5000")
        self.port_edit.setFixedWidth(70)
        conn_layout.addWidget(self.port_edit)
        self.connect_btn = QPushButton("连接")
        self.connect_btn.setFixedWidth(80)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addStretch(1)
        root.addLayout(conn_layout)

        # ---- 目标选择 ----
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("目标设备:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(300)
        target_layout.addWidget(self.target_combo)
        target_layout.addStretch(1)
        root.addLayout(target_layout)

        # ---- 命令按钮（2行×5列）----
        btn_grid = QGridLayout()
        btn_grid.setSpacing(8)
        for cfg in BUTTON_CONFIGS:
            btn = QPushButton(cfg["name"])
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda checked, c=cfg: self._on_command(c["id"], c["name"]))
            row, col = divmod(cfg["id"] - 1, 5)
            btn_grid.addWidget(btn, row, col)
        root.addLayout(btn_grid)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("未连接")

    def _connect_signals(self):
        self.connect_btn.clicked.connect(self._on_connect)
        bridge.client_list_updated.connect(self._on_client_list)
        bridge.status_changed.connect(self.statusBar().showMessage)

    # ---- 槽 ----

    def _on_connect(self):
        ip = self.ip_edit.text().strip()
        port = self.port_edit.text().strip()
        if not ip or not port:
            self.statusBar().showMessage("请填写 IP 和 Port")
            return
        url = f"ws://{ip}:{port}"
        t = threading.Thread(target=_start_ws_thread, args=(url,), daemon=True)
        t.start()

    def _on_client_list(self, clients: list):
        self._detection_clients = clients
        self.target_combo.clear()
        for c in clients:
            label = f"{c.get('device_name', '')} [{c.get('device_id', '')}]  {c.get('connection_id', '')}"
            self.target_combo.addItem(label)
        self.statusBar().showMessage(f"检测客户端列表已更新，共 {len(clients)} 台")

    def _on_command(self, button_id: int, button_name: str):
        idx = self.target_combo.currentIndex()
        if idx < 0 or idx >= len(self._detection_clients):
            self.statusBar().showMessage("请先选择目标设备")
            return
        target_id = self._detection_clients[idx]["connection_id"]
        send_command(target_id, button_id, button_name)
        self.statusBar().showMessage(f"已发送 [{button_name}] → {target_id}")


# ============================================================================
# 入口
# ============================================================================

def main():
    app = QApplication(sys.argv)
    win = ControlWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
