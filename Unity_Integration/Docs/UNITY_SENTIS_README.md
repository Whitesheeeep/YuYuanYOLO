# YuYuan 目标检测 - Unity Sentis 集成指南

## 📦 文件说明

### 模型文件

- `best.onnx` - 训练好的 YOLOv8 模型（11.7 MB）
- 位置: `E:\Master\ultralytics-main\runs\detect\runs\train\yuyuan_exp\weights\best.onnx`

### Unity 脚本

1. **YuYuanDetector.cs** - 核心检测器（使用 Sentis）
2. **YuYuanDetectorExample.cs** - 使用示例和可视化

## 🚀 Unity 集成步骤

### 1. 安装 Unity Sentis

```
Unity → Window → Package Manager → 搜索 "Sentis" → Install
```

或在 `Packages/manifest.json` 中添加：

```json
{
  "dependencies": {
    "com.unity.sentis": "1.4.0"
  }
}
```

### 2. 导入模型

1. 将 `best.onnx` 复制到 Unity 项目的 `Assets/Models/` 文件夹
2. Unity 会自动识别为 Sentis 模型资源

### 3. 导入脚本

1. 将以下脚本复制到 `Assets/Scripts/` 文件夹：
   - `YuYuanDetector.cs`
   - `YuYuanDetectorExample.cs`

### 4. 设置场景

#### 方法 A：图像检测

1. 创建空 GameObject，命名为 "YuYuanDetector"
2. 添加 `YuYuanDetector` 组件
3. 添加 `YuYuanDetectorExample` 组件
4. 在 Inspector 中：
   - 分配 `Model Asset` (best.onnx)
   - 分配 `Class Names File` (classes.txt)
   - 分配 `Test Image` (测试图片)
   - 设置 `Backend Type` (推荐 GPUCompute)
5. 运行游戏，按 **空格键** 进行检测

#### 方法 B：摄像头实时检测

1. 同上创建 GameObject 和组件
2. 在 `YuYuanDetectorExample` 中：
   - 勾选 `Use Webcam`
3. 运行游戏，按 **空格键** 捕获并检测

## 💻 代码使用示例

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
using System.Collections;

public class RealtimeDetector : MonoBehaviour
{
    public YuYuanDetector detector;
    private WebCamTexture webcam;
    private Texture2D frame;

    void Start()
    {
        webcam = new WebCamTexture(640, 480, 30);
        webcam.Play();
        frame = new Texture2D(640, 480);

        StartCoroutine(DetectLoop());
    }

    IEnumerator DetectLoop()
    {
        while (true)
        {
            // 捕获帧
            frame.SetPixels(webcam.GetPixels());
            frame.Apply();

            // 检测
            var detections = detector.Detect(frame);

            // 处理结果
            ProcessDetections(detections);

            // 等待下一帧（30 FPS）
            yield return new WaitForSeconds(1f / 30f);
        }
    }

    void ProcessDetections(List<Detection> detections)
    {
        foreach (var det in detections)
        {
            // 根据检测结果执行游戏逻辑
            switch (det.className)
            {
                case "Stone1":
                    // 处理 Stone1
                    break;
                case "LionLeft":
                    // 处理 LionLeft
                    break;
                // ... 其他类别
            }
        }
    }
}
```

### 绘制检测框

```csharp
void OnGUI()
{
    foreach (var det in detections)
    {
        // 绘制边界框
        Rect rect = det.GetRect();
        GUI.Box(rect, "");

        // 绘制标签
        string label = $"{det.className} {det.confidence:F2}";
        GUI.Label(new Rect(rect.x, rect.y - 20, 200, 20), label);
    }
}
```

## ⚙️ 参数调整

### 检测器参数

- **Confidence Threshold** (0.1-1.0): 置信度阈值，越高越严格
  - 推荐: 0.25
- **IOU Threshold** (0.1-1.0): NMS 阈值，控制重叠检测
  - 推荐: 0.45
- **Backend Type**:
  - `GPUCompute`: GPU 加速（推荐，最快）
  - `CPU`: CPU 运行（兼容性最好）

### 性能优化

1. **降低检测频率**: 不需要每帧都检测

   ```csharp
   // 每 5 帧检测一次
   if (Time.frameCount % 5 == 0)
   {
       Detect();
   }
   ```

2. **使用较小的输入尺寸**: 修改 `INPUT_SIZE` (当前 640)
   - 注意：需要重新导出 ONNX 模型

3. **异步检测**: 使用协程避免卡顿
   ```csharp
   StartCoroutine(DetectAsync());
   ```

## 🎯 检测类别

| ID  | 类别名称  | 说明     |
| --- | --------- | -------- |
| 0   | Stone1    | 石头1    |
| 1   | Picture   | 图片     |
| 2   | LionLeft  | 左侧狮子 |
| 3   | LionRight | 右侧狮子 |
| 4   | Stone2    | 石头2    |

## 🐛 常见问题

### 1. 模型加载失败

- 确保 ONNX 文件已导入到 Assets 文件夹
- 检查 Sentis 包是否正确安装

### 2. 检测结果为空

- 降低 `confidenceThreshold`
- 确保输入图像包含目标物体
- 检查图像预处理是否正确

### 3. 性能问题

- 使用 `GPUCompute` 后端
- 降低检测频率
- 减小输入图像尺寸

### 4. 坐标不准确

- 确保输入图像尺寸正确
- 检查坐标转换逻辑

## 📊 性能参考

- **GPU (RTX 3060)**: ~10-15ms/帧
- **CPU (i5-12600K)**: ~50-100ms/帧
- **模型大小**: 11.7 MB
- **输入尺寸**: 640×640

## 🔗 相关资源

- Unity Sentis 文档: https://docs.unity3d.com/Packages/com.unity.sentis@latest
- YOLOv8 文档: https://docs.ultralytics.com/
- ONNX 格式: https://onnx.ai/

## 📝 注意事项

1. **Sentis vs Barracuda**: Sentis 是新版本，性能更好，推荐使用
2. **输入格式**: 模型期望 RGB 图像，归一化到 [0, 1]
3. **坐标系**: Unity 的 Y 轴向下，可能需要转换
4. **线程安全**: Sentis Worker 不是线程安全的，在主线程使用

## ✅ 完成！

现在你可以在 Unity 中使用 YuYuan 检测器了！
