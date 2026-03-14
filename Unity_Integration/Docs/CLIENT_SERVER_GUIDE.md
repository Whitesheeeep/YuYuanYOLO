# YuYuan 客户端-服务器架构使用指南

## 🏗️ 架构说明

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│                 │  ───────────────────>  │                 │
│  Unity 客户端   │   发送图像 (Base64)    │  Python 服务器  │
│                 │  <───────────────────  │   YOLO 检测     │
│                 │   返回检测结果 (JSON)   │                 │
└─────────────────┘                        └─────────────────┘
```

### 优势
- ✅ **集中管理**: 模型在服务器端，易于更新
- ✅ **性能优化**: 服务器可以使用强大的 GPU
- ✅ **多客户端**: 多个 Unity 客户端可以共享同一服务器
- ✅ **跨平台**: Unity 客户端可以在任何平台运行

## 🖥️ Python 服务器端

### 1. 安装依赖

```bash
cd E:\Master\ultralytics-main
pip install -r server_requirements.txt
```

或手动安装：
```bash
pip install flask flask-cors ultralytics opencv-python pillow numpy
```

### 2. 启动服务器

```bash
python detection_server.py
```

输出：
```
正在加载模型...
✓ 模型加载成功

============================================================
YuYuan 检测服务器启动
============================================================
服务器地址: http://localhost:5000
检测接口: POST http://localhost:5000/detect
健康检查: GET http://localhost:5000/health
============================================================

 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.100:5000
```

### 3. API 接口

#### 健康检查
```http
GET http://localhost:5000/health
```

响应：
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["Stone1", "Picture", "LionLeft", "LionRight", "Stone2"]
}
```

#### 图像检测
```http
POST http://localhost:5000/detect
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "confidence": 0.25
}
```

响应：
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "Stone1",
      "confidence": 0.85,
      "bbox": {
        "x1": 100, "y1": 200,
        "x2": 300, "y2": 400
      }
    }
  ],
  "count": 1,
  "inference_time": 0.032,
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

### 4. 测试服务器

使用 Python 测试：
```python
import requests
import base64

# 读取图像
with open('test.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 发送请求
response = requests.post('http://localhost:5000/detect', json={
    'image': image_data,
    'confidence': 0.25
})

# 打印结果
print(response.json())
```

## 🎮 Unity 客户端

### 1. 导入脚本

将以下脚本复制到 `Assets/Scripts/`：
- `YuYuanDetectionClient.cs`
- `YuYuanClientExample.cs`

### 2. 设置场景

1. 创建空 GameObject，命名为 "DetectionClient"
2. 添加 `YuYuanDetectionClient` 组件
3. 添加 `YuYuanClientExample` 组件
4. 在 Inspector 中配置：
   - **Server Url**: `http://localhost:5000`（本地）或 `http://服务器IP:5000`（远程）
   - **Confidence Threshold**: `0.25`
   - **Test Image**: 拖入测试图片

### 3. 使用方法

#### 方法 A: 使用示例脚本
```csharp
// 运行游戏
// 按空格键: 检测图像
// 按 H 键: 检查服务器健康
```

#### 方法 B: 代码调用
```csharp
using UnityEngine;

public class MyDetector : MonoBehaviour
{
    public YuYuanDetectionClient client;
    public Texture2D image;

    void Start()
    {
        // 订阅事件
        client.OnDetectionComplete += OnDetectionComplete;
        client.OnDetectionError += OnDetectionError;

        // 检测图像
        client.DetectImage(image);
    }

    void OnDetectionComplete(DetectionResponse response)
    {
        Debug.Log($"检测到 {response.count} 个目标");

        foreach (var det in response.detections)
        {
            Debug.Log($"{det.class_name}: {det.confidence:F2}");

            // 获取边界框
            Rect rect = det.GetRect();
            Debug.Log($"位置: {rect}");
        }
    }

    void OnDetectionError(string error)
    {
        Debug.LogError($"检测失败: {error}");
    }
}
```

### 4. 实时摄像头检测

```csharp
using UnityEngine;
using System.Collections;

public class RealtimeDetector : MonoBehaviour
{
    public YuYuanDetectionClient client;
    private WebCamTexture webcam;
    private Texture2D frame;
    private bool isDetecting = false;

    void Start()
    {
        // 启动摄像头
        webcam = new WebCamTexture(640, 480, 30);
        webcam.Play();
        frame = new Texture2D(640, 480);

        // 订阅事件
        client.OnDetectionComplete += OnDetectionComplete;

        // 开始检测循环
        StartCoroutine(DetectionLoop());
    }

    IEnumerator DetectionLoop()
    {
        while (true)
        {
            if (!isDetecting)
            {
                // 捕获帧
                frame.SetPixels(webcam.GetPixels());
                frame.Apply();

                // 发送检测
                isDetecting = true;
                client.DetectImage(frame);
            }

            // 等待 100ms（10 FPS）
            yield return new WaitForSeconds(0.1f);
        }
    }

    void OnDetectionComplete(DetectionResponse response)
    {
        // 处理结果
        foreach (var det in response.detections)
        {
            Debug.Log($"{det.class_name}: {det.confidence:F2}");
        }

        isDetecting = false;
    }
}
```

## 🌐 网络配置

### 本地测试（同一台电脑）
```csharp
serverUrl = "http://localhost:5000";
```

### 局域网（不同电脑）
```csharp
// 服务器 IP: 192.168.1.100
serverUrl = "http://192.168.1.100:5000";
```

**注意**: 确保防火墙允许 5000 端口

### 查看服务器 IP
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

## ⚙️ 性能优化

### 1. 图像压缩

在发送前压缩图像：
```csharp
// 降低分辨率
Texture2D compressed = ResizeTexture(original, 640, 480);

// 使用 JPEG 压缩（更小的文件）
byte[] imageBytes = compressed.EncodeToJPG(75); // 质量 75
```

### 2. 异步检测

避免阻塞主线程：
```csharp
// 使用协程
StartCoroutine(client.DetectImageCoroutine(image));

// 或使用事件回调
client.OnDetectionComplete += HandleResult;
```

### 3. 批量检测

如果需要检测多张图像，使用批量接口：
```python
# 服务器端已实现
@app.route('/detect_batch', methods=['POST'])
```

### 4. 调整检测频率

不需要每帧都检测：
```csharp
// 每 5 帧检测一次
if (Time.frameCount % 5 == 0)
{
    client.DetectImage(frame);
}
```

## 🐛 常见问题

### Q1: 连接失败
**A**:
1. 检查服务器是否启动
2. 检查 URL 是否正确
3. 检查防火墙设置
4. 使用 `client.CheckServerHealth()` 测试连接

### Q2: 检测速度慢
**A**:
1. 降低图像分辨率
2. 使用 GPU 加速（服务器端）
3. 降低检测频率
4. 使用图像压缩

### Q3: 跨域错误（CORS）
**A**: 服务器已启用 CORS，如果仍有问题：
```python
# 在 detection_server.py 中
CORS(app, resources={r"/*": {"origins": "*"}})
```

### Q4: 超时错误
**A**: 增加超时时间：
```csharp
client.timeoutSeconds = 60; // 60 秒
```

### Q5: Base64 编码错误
**A**: 确保图像格式正确：
```csharp
// 使用 PNG
byte[] imageBytes = image.EncodeToPNG();

// 或使用 JPG（更小）
byte[] imageBytes = image.EncodeToJPG(90);
```

## 📊 性能参考

| 配置 | 检测速度 | 网络延迟 | 总时间 |
|------|---------|---------|--------|
| 本地 (localhost) | 30ms | 1ms | ~31ms |
| 局域网 (1Gbps) | 30ms | 5ms | ~35ms |
| 局域网 (100Mbps) | 30ms | 20ms | ~50ms |

## 🔒 安全建议

### 1. 添加认证

```python
# 服务器端
from flask import request

API_KEY = "your_secret_key"

@app.before_request
def check_auth():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
```

```csharp
// Unity 客户端
www.SetRequestHeader("X-API-Key", "your_secret_key");
```

### 2. 使用 HTTPS

```python
# 使用 SSL 证书
app.run(ssl_context=('cert.pem', 'key.pem'))
```

### 3. 限制请求频率

```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per minute"])
```

## ✅ 完整工作流程

1. **启动服务器**:
   ```bash
   python detection_server.py
   ```

2. **配置 Unity 客户端**:
   - 设置服务器 URL
   - 添加测试图像

3. **运行 Unity**:
   - 按空格键检测
   - 查看 Console 输出

4. **查看结果**:
   - Unity 显示检测框
   - Console 显示详细信息

## 📝 总结

- **服务器端**: Python + Flask + YOLO
- **客户端**: Unity + UnityWebRequest
- **通信**: HTTP + JSON + Base64
- **优势**: 集中管理、高性能、易扩展

现在你可以在 Unity 中通过网络调用 Python 服务器进行目标检测了！
