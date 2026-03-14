# YuYuan 检测监控系统 - 使用说明

## 系统简介

这是一个集成的 YOLO 检测监控系统，将 WebSocket 服务器和 PyQt 监控界面合并到一个程序中。

## 文件说明

- `detection_monitor.py` - 主程序（包含服务器和监控界面）
- `Unity_Integration/Scripts/YuYuanWebSocketClient.cs` - Unity 客户端脚本

## 快速启动

### 1. 启动监控系统

只需运行一个命令：

```bash
cd E:\Master\ultralytics-main
python detection_monitor.py
```

这将同时启动：

- WebSocket 服务器（监听 ws://0.0.0.0:5000）
- PyQt 监控界面窗口

### 2. 连接 Unity 客户端

在 Unity 中：

1. 安装 NativeWebSocket 包：
   - Window → Package Manager → + → Add package from git URL
   - 输入：`https://github.com/endel/NativeWebSocket.git`

2. 将 `YuYuanWebSocketClient.cs` 脚本添加到场景中的 GameObject

3. 配置脚本参数：
   - Server URL: `ws://localhost:5000`（本地测试）或 `ws://服务器IP:5000`（远程连接）
   - Device ID: 设备唯一标识（如 `device_001`）
   - Device Name: 设备显示名称（如 `Android Device 1`）

4. 运行场景

## 功能特点

### 监控界面功能

- ✅ 实时显示所有连接的设备
- ✅ 双窗口显示：原始视频流 + 检测后视频流
- ✅ 设备列表：绿色圆点表示在线设备
- ✅ 点击设备可切换显示
- ✅ 状态栏显示当前设备和检测数量

### WebSocket 服务器功能

- ✅ 接收来自多个 Unity 客户端的图像
- ✅ 使用 YOLO 进行实时检测
- ✅ 绘制检测框和标签
- ✅ 广播结果到所有连接的客户端

## 系统架构

```
┌─────────────────────────────────────────┐
│     detection_monitor.py                │
│  ┌────────────────┐  ┌────────────────┐ │
│  │ WebSocket 服务器│  │ PyQt 监控界面  │ │
│  │ (后台线程)      │  │ (主线程)       │ │
│  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────┘
              ↕ WebSocket
┌─────────────────────────────────────────┐
│     Unity 客户端（多个设备）             │
│  ┌──────────┐  ┌──────────┐            │
│  │ 设备 1   │  │ 设备 2   │  ...       │
│  └──────────┘  └──────────┘            │
└─────────────────────────────────────────┘
```

## 依赖要求

确保已安装所有依赖：

```bash
pip install websockets ultralytics opencv-python pillow numpy PyQt5
```

## 常见问题

### Q: 监控界面无法连接到服务器

A: 确保程序已完全启动，等待 1-2 秒后监控界面会自动连接

### Q: Unity 客户端无法连接

A: 检查：

1. 服务器 IP 地址是否正确
2. 防火墙是否允许端口 5000
3. 网络连接是否正常

### Q: 检测速度慢

A: 可以调整：

1. Unity 客户端的 FPS 设置（降低帧率）
2. 图像质量设置（降低 JPEG 质量）
3. 图像分辨率（降低 targetWidth 和 targetHeight）

## 技术细节

### 消息协议

**Unity → 服务器（检测请求）**:

```json
{
  "type": "detect",
  "device_id": "device_001",
  "device_name": "Android Device 1",
  "image": "base64_encoded_image"
}
```

**服务器 → 所有客户端（检测结果）**:

```json
{
  "type": "detection_result",
  "device_id": "device_001",
  "device_name": "Android Device 1",
  "original_image": "base64_encoded",
  "detected_image": "base64_encoded_with_boxes",
  "detections": [
    {
      "class": "Stone1",
      "confidence": 0.85,
      "bbox": [100, 200, 300, 400]
    }
  ]
}
```

## 优势

1. **一键启动** - 无需分别启动服务器和监控界面
2. **极简架构** - 代码简洁，易于理解和维护
3. **实时监控** - WebSocket 双向通信，延迟 < 50ms
4. **多设备支持** - 可同时连接多个 Unity 客户端
5. **自动管理** - 设备连接/断开自动更新界面

## 开发者信息

- YOLO 模型路径：`E:\Master\ultralytics-main\runs\detect\runs\train\yuyuan_exp\weights\best.pt`
- WebSocket 端口：5000
- 支持的图像格式：JPEG（base64 编码）
