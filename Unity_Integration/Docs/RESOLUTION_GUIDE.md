# Unity 输入分辨率说明

## 📐 分辨率要求总结

### 模型要求

- **固定输入**: 640×640 像素
- **自动调整**: 脚本会自动将任意分辨率调整为 640×640

### 实际使用

✅ **可以输入任意分辨率的图像**

- 1920×1080 ✅
- 1280×720 ✅
- 800×600 ✅
- 640×480 ✅
- 任意尺寸 ✅

## 🎨 三种调整模式

### 1. Letterbox（推荐）⭐

**特点**: 保持宽高比，添加黑边

```
输入: 1920×1080 (16:9)
     ┌─────────────────┐
     │                 │
     │   原始图像      │
     │                 │
     └─────────────────┘

输出: 640×640 (1:1)
     ┌─────────────────┐
     │█████████████████│ ← 黑边
     │                 │
     │   原始图像      │
     │                 │
     │█████████████████│ ← 黑边
     └─────────────────┘
```

**优点**:

- ✅ 不变形，检测精度最高
- ✅ 保持原始宽高比
- ✅ 适合所有场景

**缺点**:

- ⚠️ 有效检测区域略小（因为有黑边）

**使用场景**: 默认推荐，适合大多数情况

### 2. Stretch（拉伸）

**特点**: 直接拉伸到目标尺寸

```
输入: 1920×1080 (16:9)
     ┌─────────────────┐
     │                 │
     │   原始图像      │
     │                 │
     └─────────────────┘

输出: 640×640 (1:1)
     ┌───────────┐
     │           │
     │ 拉伸变形  │
     │           │
     │           │
     └───────────┘
```

**优点**:

- ✅ 使用全部 640×640 区域
- ✅ 处理速度最快

**缺点**:

- ❌ 图像变形，可能降低检测精度
- ❌ 宽高比改变

**使用场景**: 输入图像已经是 1:1 宽高比时

### 3. CenterCrop（中心裁剪）

**特点**: 裁剪中心区域

```
输入: 1920×1080 (16:9)
     ┌─────────────────┐
     │ ✂️ │         │ ✂️ │
     │   │ 保留区域│   │
     │ ✂️ │         │ ✂️ │
     └─────────────────┘

输出: 640×640 (1:1)
     ┌───────────┐
     │           │
     │ 中心区域  │
     │           │
     └───────────┘
```

**优点**:

- ✅ 不变形
- ✅ 使用全部 640×640 区域

**缺点**:

- ❌ 边缘内容被裁剪
- ❌ 可能丢失重要目标

**使用场景**: 目标物体总是在图像中心时

## ⚙️ 在 Unity 中配置

### Inspector 设置

```
YuYuanDetector
├── Model Asset: best.onnx
├── Class Names File: classes.txt
├── Confidence Threshold: 0.25
├── IOU Threshold: 0.45
├── Resize Mode: Letterbox ← 选择调整模式
└── Backend Type: GPUCompute
```

### 代码设置

```csharp
YuYuanDetector detector = GetComponent<YuYuanDetector>();

// 设置调整模式
detector.resizeMode = YuYuanDetector.ResizeMode.Letterbox;  // 推荐
// detector.resizeMode = YuYuanDetector.ResizeMode.Stretch;
// detector.resizeMode = YuYuanDetector.ResizeMode.CenterCrop;
```

## 📊 性能对比

| 模式       | 处理速度 | 检测精度   | 推荐度     |
| ---------- | -------- | ---------- | ---------- |
| Letterbox  | 中等     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Stretch    | 最快     | ⭐⭐⭐     | ⭐⭐       |
| CenterCrop | 中等     | ⭐⭐⭐⭐   | ⭐⭐⭐     |

## 🎯 不同场景推荐

### 场景 1: 通用目标检测

```csharp
detector.resizeMode = YuYuanDetector.ResizeMode.Letterbox;
```

**原因**: 保持宽高比，检测精度最高

### 场景 2: 正方形图像输入

```csharp
detector.resizeMode = YuYuanDetector.ResizeMode.Stretch;
```

**原因**: 输入已经是 1:1，拉伸不会变形

### 场景 3: 中心区域检测

```csharp
detector.resizeMode = YuYuanDetector.ResizeMode.CenterCrop;
```

**原因**: 目标总在中心，裁剪不影响检测

### 场景 4: 实时摄像头

```csharp
detector.resizeMode = YuYuanDetector.ResizeMode.Letterbox;
```

**原因**: 摄像头通常是 16:9，需要保持宽高比

## 💡 最佳实践

### 1. 预处理输入图像

如果可能，在传入检测器之前就调整为 640×640：

```csharp
// 方法 1: 使用正方形摄像头
WebCamTexture webcam = new WebCamTexture(640, 640, 30);

// 方法 2: 预先裁剪图像
Texture2D square = CropToSquare(originalImage);
detector.Detect(square);
```

### 2. 批量检测时统一尺寸

```csharp
// 如果检测多张图像，统一调整为相同尺寸
List<Texture2D> images = GetImages();
foreach (var img in images)
{
    // 所有图像使用相同的调整模式
    var detections = detector.Detect(img);
}
```

### 3. 性能优化

```csharp
// 降低输入分辨率可以提高速度
// 但需要重新导出模型
// 例如: 320×320 会比 640×640 快 4 倍
```

## 🔧 修改模型输入尺寸

如果需要使用不同的输入尺寸（如 320×320 或 1280×1280）：

### 1. 重新导出模型

```python
# 在 Python 中重新导出
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="onnx",
    imgsz=320,  # 修改为 320×320
    simplify=True,
    opset=12,
)
```

### 2. 修改 Unity 脚本

```csharp
// YuYuanDetector.cs
private const int INPUT_SIZE = 320;  // 改为 320
```

### 尺寸选择建议

| 尺寸      | 速度   | 精度       | 适用场景           |
| --------- | ------ | ---------- | ------------------ |
| 320×320   | 🚀🚀🚀 | ⭐⭐       | 实时检测，移动设备 |
| 640×640   | 🚀🚀   | ⭐⭐⭐⭐   | 平衡（推荐）       |
| 1280×1280 | 🚀     | ⭐⭐⭐⭐⭐ | 高精度，小目标检测 |

## ❓ 常见问题

### Q1: 为什么检测结果不准确？

**A**: 可能是图像被拉伸变形，尝试使用 `Letterbox` 模式

### Q2: 可以使用 1920×1080 的图像吗？

**A**: 可以，脚本会自动调整为 640×640

### Q3: 如何提高检测速度？

**A**:

1. 使用 `Stretch` 模式（最快）
2. 降低输入图像分辨率
3. 使用 GPU 后端
4. 重新导出更小尺寸的模型（如 320×320）

### Q4: Letterbox 模式的黑边会影响检测吗？

**A**: 不会，黑边区域不会产生误检测

### Q5: 可以动态切换调整模式吗？

**A**: 可以，在运行时修改 `detector.resizeMode` 即可

## ✅ 总结

- **模型要求**: 固定 640×640 输入
- **实际使用**: 任意分辨率都可以
- **推荐模式**: Letterbox（保持宽高比）
- **性能优化**: 根据场景选择合适的模式

使用 `Letterbox` 模式可以获得最佳的检测精度！
