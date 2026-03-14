# YuYuan 目标检测 - Unity Sentis 集成包

这个文件夹包含了将 YuYuan YOLO 模型集成到 Unity 中所需的所有文件。

## 📦 文件结构

```
Unity_Integration/
├── Scripts/                    # Unity C# 脚本
│   ├── YuYuanDetector.cs      # 核心检测器（Sentis）
│   ├── YuYuanDetectorExample.cs # 使用示例
│   └── CameraMgr.cs           # 摄像头采集管理
├── Models/                     # 模型文件
│   └── best.onnx              # YOLOv8 ONNX 模型（11.7 MB）
├── Config/                     # 配置文件
│   └── classes.txt            # 类别名称配置
├── Docs/                       # 文档
│   ├── UNITY_SENTIS_README.md # Unity 集成指南
│   └── CLASSES_CONFIG_README.md # 类别配置说明
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 安装 Unity Sentis

在 Unity 中打开 Package Manager：

```
Window → Package Manager → 搜索 "Sentis" → Install
```

或在 `Packages/manifest.json` 中添加：

```json
{
  "dependencies": {
    "com.unity.sentis": "1.4.0"
  }
}
```

### 2. 导入文件到 Unity 项目

将文件复制到 Unity 项目的对应位置：

```
Unity 项目/
├── Assets/
│   ├── Scripts/
│   │   ├── YuYuanDetector.cs           ← 从 Scripts/ 复制
│   │   ├── YuYuanDetectorExample.cs    ← 从 Scripts/ 复制
│   │   └── CameraMgr.cs                ← 从 Scripts/ 复制
│   ├── Models/
│   │   └── best.onnx                   ← 从 Models/ 复制
│   └── Resources/
│       └── classes.txt                 ← 从 Config/ 复制
```

### 3. 设置场景

1. 创建空 GameObject，命名为 "YuYuanDetector"
2. 添加 `YuYuanDetector` 组件
3. 添加 `YuYuanDetectorExample` 组件
4. 如果要启用摄像头检测，添加 `CameraMgr` 组件
5. 在 Inspector 中配置：
   - **Model Asset**: 拖入 `best.onnx`
   - **Class Names File**: 拖入 `classes.txt`
   - **Test Image**: 拖入测试图片（可选）
   - **Backend Type**: 选择 `GPUCompute`（推荐）
   - **Use Camera**: 勾选启用摄像头
   - **CameraMgr**: 拖入同一物体上的 `CameraMgr`

### 4. 运行测试

- 运行游戏
- 按 **空格键** 进行检测
- 查看 Console 输出和可视化结果

## 🎯 检测类别

模型可以检测以下 5 个类别：

| ID  | 类别名称  | 说明     |
| --- | --------- | -------- |
| 0   | Stone1    | 石头1    |
| 1   | Picture   | 图片     |
| 2   | LionLeft  | 左侧狮子 |
| 3   | LionRight | 右侧狮子 |
| 4   | Stone2    | 石头2    |

## 📖 详细文档

- **Unity 集成指南**: `Docs/UNITY_SENTIS_README.md`
  - 完整的集成步骤
  - 代码使用示例
  - 性能优化建议
  - 常见问题解答

- **类别配置说明**: `Docs/CLASSES_CONFIG_README.md`
  - 如何修改类别名称
  - 多语言支持
  - 配置文件格式

## 💻 代码示例

### 基础检测

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MyDetector : MonoBehaviour
{
    public YuYuanDetector detector;
    public Texture2D image;

    void Start()
    {
        // 检测图像
        List<Detection> results = detector.Detect(image);

        // 处理结果
        foreach (var det in results)
        {
            Debug.Log($"检测到: {det.className}");
            Debug.Log($"置信度: {det.confidence:F2}");
            Debug.Log($"位置: {det.GetRect()}");
        }
    }
}
```

### 实时摄像头检测

```csharp
using UnityEngine;

public class RealtimeDetector : MonoBehaviour
{
    public YuYuanDetector detector;
    public CameraMgr cameraMgr;

    void Start()
    {
        if (cameraMgr != null)
        {
            cameraMgr.cropToTarget = false;
            cameraMgr.StartCamera();
        }
    }

    void Update()
    {
        if (cameraMgr == null || detector == null)
            return;

        var frame = cameraMgr.FrameTexture;
        if (frame != null)
        {
            var detections = detector.Detect(frame);
            foreach (var det in detections)
            {
                Debug.Log($"{det.className}: {det.confidence:F2}");
            }
        }
    }
}
```

## ⚙️ 参数配置

### 检测器参数

- **Confidence Threshold** (0.1-1.0): 置信度阈值
  - 默认: 0.25
  - 越高越严格，检测结果越少但更准确

- **IOU Threshold** (0.1-1.0): NMS 阈值
  - 默认: 0.45
  - 控制重叠检测的过滤

- **Backend Type**: 运行后端
  - `GPUCompute`: GPU 加速（推荐，最快）
  - `CPU`: CPU 运行（兼容性最好）

## 📊 性能参考

- **GPU (RTX 3060)**: ~10-15ms/帧
- **CPU (i5-12600K)**: ~50-100ms/帧
- **模型大小**: 11.7 MB
- **输入尺寸**: 640×640

## 🔧 自定义类别名称

编辑 `Config/classes.txt` 文件即可修改类别名称：

```
石头1
图片
左侧狮子
右侧狮子
石头2
```

无需重新编译代码，重启游戏即可生效。

## 🐛 常见问题

### 1. 模型加载失败

- 确保 ONNX 文件在 `Assets/Models/` 文件夹
- 检查 Sentis 包是否正确安装

### 2. 检测结果为空

- 降低 `confidenceThreshold`
- 确保输入图像包含目标物体

### 3. 性能问题

- 使用 `GPUCompute` 后端
- 降低检测频率（不需要每帧检测）

### 4. 类别名称不显示

- 确保 `classes.txt` 在 `Assets/Resources/` 文件夹
- 在 Inspector 中分配 `Class Names File`

## 📝 模型信息

- **模型类型**: YOLOv8n (Nano)
- **训练数据**: YuYuan 自定义数据集
- **输入格式**: RGB 图像，640×640
- **输出格式**: [1, 9, 8400]
  - 9 = 4 (bbox) + 5 (classes)
  - 8400 = 检测点数量

## 🔗 相关资源

- Unity Sentis 文档: https://docs.unity3d.com/Packages/com.unity.sentis@latest
- YOLOv8 文档: https://docs.ultralytics.com/
- ONNX 格式: https://onnx.ai/

## 📄 许可证

本项目使用 AGPL-3.0 许可证（继承自 Ultralytics YOLO）

## ✅ 完成

现在你可以在 Unity 中使用 YuYuan 检测器进行实时目标检测了！

如有问题，请查看 `Docs/` 文件夹中的详细文档。
