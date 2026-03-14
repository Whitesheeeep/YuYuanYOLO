using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Linq;

public class YuYuanDetector : MonoBehaviour
{
    [Header("模型设置")]
    public ModelAsset modelAsset;

    [Header("类别设置")]
    [Tooltip("类别名称文件 (TextAsset)，每行一个类别名称")]
    public TextAsset classNamesFile;

    [Header("检测参数")]
    [Range(0.1f, 1f)]
    public float confidenceThreshold = 0.25f;

    [Range(0.1f, 1f)]
    public float iouThreshold = 0.45f;

    [Header("图像预处理")]
    [Tooltip("图像调整模式")]
    public ResizeMode resizeMode = ResizeMode.Letterbox;

    [Header("运行时设置")]
    public BackendType backendType = BackendType.GPUCompute;

    [Header("输入归一化")]
    public bool normalizeInput = true;

    [Header("对齐调试")]
    public bool debugAlignment = false;
    [Range(1, 50)]
    public int debugSampleCount = 10;

    // 模型参数
    private const int INPUT_SIZE = 640;
    private int NUM_CLASSES = 5;

    // letterbox 反变换参数
    private float lastLetterboxScale = 1f;
    private int lastPadX = 0;
    private int lastPadY = 0;

    // 类别名称
    private string[] classNames;

    /// <summary>
    /// 图像调整模式
    /// </summary>
    public enum ResizeMode
    {
        Letterbox,      // 保持宽高比，添加黑边（推荐）
        Stretch,        // 拉伸到目标尺寸
        CenterCrop      // 裁剪中心区域
    }

    // Sentis 运行时
    private Model runtimeModel;
    private Worker worker;

    void Start()
    {
        InitializeModel();
    }

    void InitializeModel()
    {
        // 加载类别名称
        LoadClassNames();

        if (modelAsset == null)
        {
            Debug.LogError("模型资源未分配！请在 Inspector 中分配 ONNX 模型。");
            return;
        }

        // 加载模型
        runtimeModel = ModelLoader.Load(modelAsset);

        // 创建 Worker
        worker = new Worker(runtimeModel, backendType);

        Debug.Log($"YuYuan 检测器初始化成功 - 后端: {backendType}, 类别数: {NUM_CLASSES}");
    }

    /// <summary>
    /// 加载类别名称
    /// </summary>
    private void LoadClassNames()
    {
        if (classNamesFile != null)
        {
            // 从 TextAsset 加载
            string[] lines = classNamesFile.text.Split('\n');
            List<string> names = new List<string>();

            foreach (string line in lines)
            {
                string trimmed = line.Trim();
                if (!string.IsNullOrEmpty(trimmed))
                {
                    names.Add(trimmed);
                }
            }

            classNames = names.ToArray();
            NUM_CLASSES = classNames.Length;

            Debug.Log($"从文件加载了 {NUM_CLASSES} 个类别:");
            for (int i = 0; i < classNames.Length; i++)
            {
                Debug.Log($"  [{i}] {classNames[i]}");
            }
        }
        else
        {
            // 使用默认类别名称
            Debug.LogWarning("未分配类别文件，使用默认类别名称");
            classNames = new[]
            {
                "Stone1", "Picture", "LionLeft", "LionRight", "Stone2"
            };
            NUM_CLASSES = classNames.Length;
        }
    }

    /// <summary>
    /// 检测图像中的目标
    /// </summary>
    public List<Detection> Detect(Texture2D image)
    {
        if (worker == null)
        {
            Debug.LogError("模型未初始化！");
            return new List<Detection>();
        }

        if (debugAlignment)
        {
            Debug.Log($"Unity input texture size: {image.width}x{image.height}");
        }

        // 1. 预处理
        using Tensor<float> inputTensor = PreprocessImage(image);

        // 2. 推理
        worker.Schedule(inputTensor);

        // 3. 获取输出并读回 CPU
        using Tensor<float> outputTensor = (worker.PeekOutput() as Tensor<float>)?.ReadbackAndClone();
        if (outputTensor == null)
        {
            Debug.LogError("输出张量类型不匹配，无法解析为 Tensor<float>。");
            return new List<Detection>();
        }

        // 4. 后处理
        List<Detection> detections = PostprocessOutput(outputTensor, image.width, image.height);

        return detections;
    }

    /// <summary>
    /// 预处理图像
    /// </summary>
    private Tensor<float> PreprocessImage(Texture2D image)
    {
        // 根据选择的模式调整图像大小
        Texture2D resized;
        switch (resizeMode)
        {
            case ResizeMode.Letterbox:
                resized = ImagePreprocessor.ResizeWithLetterbox(image, INPUT_SIZE, out lastLetterboxScale, out lastPadX, out lastPadY);
                break;
            case ResizeMode.CenterCrop:
                resized = ImagePreprocessor.ResizeCenterCrop(image, INPUT_SIZE);
                lastLetterboxScale = 1f;
                lastPadX = 0;
                lastPadY = 0;
                break;
            case ResizeMode.Stretch:
            default:
                resized = ImagePreprocessor.ResizeStretch(image, INPUT_SIZE);
                lastLetterboxScale = 1f;
                lastPadX = 0;
                lastPadY = 0;
                break;
        }

        // 转换为 Tensor
        // Sentis 使用 NCHW 格式: [batch, channels, height, width]
        var transform = new TextureTransform().SetDimensions(INPUT_SIZE, INPUT_SIZE, 3).SetTensorLayout(TensorLayout.NCHW);
        Tensor<float> tensor = new Tensor<float>(new TensorShape(1, 3, INPUT_SIZE, INPUT_SIZE));
        TextureConverter.ToTensor(resized, tensor, transform);

        if (normalizeInput)
        {
            using Tensor<float> cpuTensor = tensor.ReadbackAndClone();
            if (cpuTensor != null)
            {
                float[] data = cpuTensor.DownloadToArray();
                float rawMin = float.MaxValue;
                float rawMax = float.MinValue;

                for (int i = 0; i < data.Length; i++)
                {
                    float v = data[i];
                    if (v < rawMin) rawMin = v;
                    if (v > rawMax) rawMax = v;
                }

                bool applyNormalization = rawMax > 1.5f;
                if (applyNormalization)
                {
                    for (int i = 0; i < data.Length; i++)
                    {
                        data[i] = data[i] / 255f;
                    }
                }

                if (debugAlignment)
                {
                    int sample = Mathf.Clamp(debugSampleCount, 1, 50);
                    float[] first = data.Take(sample).ToArray();
                    Debug.Log($"Input shape={cpuTensor.shape} rawMin={rawMin:F6} rawMax={rawMax:F6} normalized={applyNormalization} first={string.Join(",", first.Select(v => v.ToString("F6")))} scale={lastLetterboxScale:F6} pad=({lastPadX},{lastPadY})");
                }

                if (applyNormalization)
                {
                    tensor.Dispose();
                    tensor = new Tensor<float>(cpuTensor.shape, data);
                }
            }
        }
        else if (debugAlignment)
        {
            using Tensor<float> cpuTensor = tensor.ReadbackAndClone();
            if (cpuTensor != null)
            {
                float[] data = cpuTensor.DownloadToArray();
                float min = data.Min();
                float max = data.Max();
                int sample = Mathf.Clamp(debugSampleCount, 1, 50);
                float[] first = data.Take(sample).ToArray();

                Debug.Log($"Input shape={cpuTensor.shape} min={min:F6} max={max:F6} first={string.Join(",", first.Select(v => v.ToString("F6")))} scale={lastLetterboxScale:F6} pad=({lastPadX},{lastPadY})");
            }
        }

        return tensor;
    }

    /// <summary>
    /// 后处理模型输出
    /// </summary>
    private List<Detection> PostprocessOutput(Tensor<float> output, int originalWidth, int originalHeight)
    {
        List<Detection> detections = new List<Detection>();

        // 兼容 [1, C, N] 或 [1, N, C]
        int dim1 = output.shape[1];
        int dim2 = output.shape[2];
        bool channelsFirst = dim1 <= dim2;
        int featureCount = channelsFirst ? dim1 : dim2;
        int numDetections = channelsFirst ? dim2 : dim1;

        int expectedNoObj = 4 + NUM_CLASSES;
        int expectedWithObj = 5 + NUM_CLASSES;
        bool hasObjectness = featureCount == expectedWithObj;

        if (featureCount != expectedNoObj && featureCount != expectedWithObj)
        {
            Debug.LogWarning($"输出特征维度 {featureCount} 与预期不一致，NUM_CLASSES={NUM_CLASSES}。");
        }

        float GetValue(int detIndex, int featIndex)
        {
            return channelsFirst ? output[0, featIndex, detIndex] : output[0, detIndex, featIndex];
        }

        if (debugAlignment)
        {
            int sample = Mathf.Clamp(debugSampleCount, 1, 50);
            sample = Mathf.Min(sample, featureCount);
            string first = string.Join(",", Enumerable.Range(0, sample).Select(i => GetValue(0, i).ToString("F6")));
            Debug.Log($"Output shape={output.shape} channelsFirst={channelsFirst} featureCount={featureCount} numDetections={numDetections} hasObj={hasObjectness} firstDet0={first}");
        }

        for (int i = 0; i < numDetections; i++)
        {
            // 获取边界框 (中心点格式)
            float cx = GetValue(i, 0);
            float cy = GetValue(i, 1);
            float w = GetValue(i, 2);
            float h = GetValue(i, 3);

            float obj = hasObjectness ? GetValue(i, 4) : 1f;
            int classOffset = hasObjectness ? 5 : 4;

            // 获取类别置信度
            float maxConf = 0f;
            int maxClass = 0;

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                float conf = GetValue(i, classOffset + c);
                if (conf > maxConf)
                {
                    maxConf = conf;
                    maxClass = c;
                }
            }

            float finalConf = maxConf * obj;

            // 过滤低置信度检测
            if (finalConf > confidenceThreshold)
            {
                float x1 = cx - w / 2f;
                float y1 = cy - h / 2f;
                float x2 = cx + w / 2f;
                float y2 = cy + h / 2f;

                if (resizeMode == ResizeMode.Letterbox)
                {
                    // 反 letterbox: 去 pad 后除以 scale
                    x1 = (x1 - lastPadX) / lastLetterboxScale;
                    y1 = (y1 - lastPadY) / lastLetterboxScale;
                    x2 = (x2 - lastPadX) / lastLetterboxScale;
                    y2 = (y2 - lastPadY) / lastLetterboxScale;
                }
                else
                {
                    float scaleX = (float)originalWidth / INPUT_SIZE;
                    float scaleY = (float)originalHeight / INPUT_SIZE;
                    x1 *= scaleX;
                    y1 *= scaleY;
                    x2 *= scaleX;
                    y2 *= scaleY;
                }

                detections.Add(new Detection
                {
                    x1 = x1,
                    y1 = y1,
                    x2 = x2,
                    y2 = y2,
                    confidence = finalConf,
                    classId = maxClass,
                    className = classNames[maxClass]
                });
            }
        }

        // 应用 NMS (非极大值抑制)
        detections = ApplyNMS(detections, iouThreshold);

        return detections;
    }

    /// <summary>
    /// 非极大值抑制
    /// </summary>
    private List<Detection> ApplyNMS(List<Detection> detections, float iouThreshold)
    {
        // 按置信度降序排序
        detections = detections.OrderByDescending(d => d.confidence).ToList();

        List<Detection> result = new List<Detection>();

        while (detections.Count > 0)
        {
            Detection best = detections[0];
            result.Add(best);
            detections.RemoveAt(0);

            // 移除与最佳检测重叠度高的检测
            detections.RemoveAll(det =>
                det.classId == best.classId &&
                CalculateIOU(best, det) > iouThreshold
            );
        }

        return result;
    }

    /// <summary>
    /// 计算 IOU (交并比)
    /// </summary>
    private float CalculateIOU(Detection a, Detection b)
    {
        float x1 = Mathf.Max(a.x1, b.x1);
        float y1 = Mathf.Max(a.y1, b.y1);
        float x2 = Mathf.Min(a.x2, b.x2);
        float y2 = Mathf.Min(a.y2, b.y2);

        float intersection = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0;
    }

    /// <summary>
    /// 调整图像大小
    /// </summary>
    private Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        RenderTexture rt = RenderTexture.GetTemporary(width, height);
        rt.filterMode = FilterMode.Bilinear;

        RenderTexture.active = rt;
        Graphics.Blit(source, rt);

        Texture2D result = new Texture2D(width, height, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        result.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        return result;
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}

/// <summary>
/// 检测结果类
/// </summary>
[System.Serializable]
public class Detection
{
    public float x1, y1, x2, y2;
    public float confidence;
    public int classId;
    public string className;

    public Rect GetRect()
    {
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    public Vector2 GetCenter()
    {
        return new Vector2((x1 + x2) / 2, (y1 + y2) / 2);
    }

    public override string ToString()
    {
        return $"{className} ({confidence:F2}) at ({x1:F0}, {y1:F0}, {x2:F0}, {y2:F0})";
    }
}
