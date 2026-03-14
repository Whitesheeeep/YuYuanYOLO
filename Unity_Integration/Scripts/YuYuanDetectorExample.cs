using UnityEngine;
using System.Collections.Generic;
using System.IO;
using UnityEngine.UI;

/// <summary>
/// YuYuan 检测器使用示例
/// 演示如何在 Unity 中使用 Sentis 进行实时目标检测
/// </summary>
public class YuYuanDetectorExample : MonoBehaviour
{
    [Header("检测器")]
    public YuYuanDetector detector;

    [Header("测试图像")]
    public Texture2D testImage;
    [Tooltip("当按下 Space 键时，从 StreamingAssets 加载 img.png 进行检测")]
    public string testImageName = "img.png";
    private string TestImagePath => testImageName +"."+ testEImageSuffix; // StreamingAssets 中的测试图像路径
    public E_ImageSuffix testEImageSuffix = E_ImageSuffix.png;
    public enum E_ImageSuffix
    {
        jpg,
        png
    }

    [Header("摄像头检测")]
    public bool useCamera = true;
    public float detectInterval = 0.2f;
    public CameraMgr cameraMgr;
    Coroutine detectRoutine;
    bool isDetecting;

    float lastInterval;

    [Header("可视化")]
    public bool drawGizmos = true;
    public int guiFontSize = 24;
    public RawImage resultImage;
    public bool updateResultImage = true;
    public bool updateDisplayEveryFrame = false;
    public int boxLineWidth = 2;
    public Color[] classColors = new Color[]
    {
        Color.red,      // Stone1
        Color.cyan,     // Picture
        Color.blue,     // LionLeft
        Color.yellow,   // LionRight
        Color.green     // Stone2
    };

    [Header("日志")]
    public bool logDetectionsToConsole = true;
    public float logIntervalSeconds = 1f;

    private List<Detection> currentDetections = new List<Detection>();
    private Texture2D lastVisualized;
    float lastLogTime;
    GUIStyle labelStyle;
    Texture2D lineTexture;

    void Start()
    {
        if (detector == null)
        {
            detector = GetComponent<YuYuanDetector>();
        }

        lastInterval = Mathf.Max(0.01f, detectInterval);
        detectInterval = lastInterval;

        if (useCamera)
        {
            if (cameraMgr == null)
                cameraMgr = GetComponent<CameraMgr>();

            if (cameraMgr != null)
            {
                // Use original frame resolution for detection accuracy.
                cameraMgr.cropToTarget = false;
                cameraMgr.StartCamera();
            }

            detectRoutine = StartCoroutine(DetectLoop());
        }

        // GUI 资源改为在 OnGUI 中懒加载，避免非 OnGUI 调用 GUI.skin
    }

    void InitializeGuiResources()
    {
        if (labelStyle == null)
        {
            labelStyle = new GUIStyle
            {
                fontSize = guiFontSize,
                normal = { textColor = Color.white }
            };
        }

        if (lineTexture == null)
        {
            lineTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            lineTexture.SetPixel(0, 0, Color.white);
            lineTexture.Apply();
        }
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            DetectFromImage();
            if (testImage != null)
            {

            }
        }

        if (useCamera && detectRoutine != null && !Mathf.Approximately(lastInterval, detectInterval))
        {
            SetDetectInterval(detectInterval);
        }

        if (useCamera && cameraMgr != null && resultImage != null)
        {
            var cameraFrame = cameraMgr.FrameTexture;
            if (cameraFrame != null)
            {
                SyncResultAspect(cameraFrame);
            }
        }

        if (useCamera && updateResultImage && updateDisplayEveryFrame && cameraMgr != null)
        {
            var displayFrame = cameraMgr.FrameTexture ?? cameraMgr.OutputTexture;
            if (displayFrame != null)
            {
                UpdateResultVisualization(displayFrame, currentDetections);
            }
        }
    }

    public void SetDetectInterval(float seconds)
    {
        detectInterval = Mathf.Max(0.01f, seconds);
        lastInterval = detectInterval;

        if (!useCamera)
            return;

        if (detectRoutine != null)
            StopCoroutine(detectRoutine);

        detectRoutine = StartCoroutine(DetectLoop());
    }

    System.Collections.IEnumerator DetectLoop()
    {
        var wait = new WaitForSeconds(detectInterval);
        while (isActiveAndEnabled)
        {
            if (!isDetecting && detector != null && cameraMgr != null)
            {
                var detectFrame = cameraMgr.FrameTexture ?? cameraMgr.OutputTexture;
                if (detectFrame != null)
                {
                    isDetecting = true;
                    currentDetections = detector.Detect(detectFrame);
                    LogDetections(currentDetections);
                    if (!updateDisplayEveryFrame)
                    {
                        UpdateResultVisualization(detectFrame, currentDetections);
                    }
                    isDetecting = false;
                }
            }

            yield return wait;
        }

        yield break;
    }

    void LogDetections(List<Detection> detections)
    {
        if (!logDetectionsToConsole)
            return;

        if (Time.time - lastLogTime < logIntervalSeconds)
            return;

        lastLogTime = Time.time;

        if (detections == null || detections.Count == 0)
            return;

        var names = new HashSet<string>();
        foreach (var det in detections)
        {
            if (!string.IsNullOrEmpty(det.className))
                names.Add(det.className);
        }

        if (names.Count > 0)
            Debug.Log($"检测到: {string.Join(", ", names)}");
    }

    /// <summary>
    /// 从测试图像检测
    /// </summary>
    public void DetectFromImage()
    {
        if (testImage == null)
        {
            Debug.LogWarning("测试图像未设置！尝试从 StreamingAssets 加载 img.jpg");
            var imgData = File.ReadAllBytes(Path.Combine(Application.streamingAssetsPath,TestImagePath));
            testImage = new Texture2D(2, 2);
            testImage.LoadImage(imgData);
        }

        Debug.Log("开始检测...");
        float startTime = Time.realtimeSinceStartup;

        currentDetections = detector.Detect(testImage);

        float elapsed = (Time.realtimeSinceStartup - startTime) * 1000f;
        Debug.Log($"检测完成！用时: {elapsed:F2}ms, 检测到 {currentDetections.Count} 个目标");

        foreach (var det in currentDetections)
        {
            Debug.Log($"  - {det}");
        }

        UpdateResultVisualization(testImage, currentDetections);
    }

    void UpdateResultVisualization(Texture2D source, List<Detection> detections)
    {
        if (!updateResultImage || resultImage == null || source == null)
            return;

        Texture2D snapshot = CopyToTexture2D(source);
        DrawDetections(snapshot, GetDisplayDetections(source, detections));

        if (lastVisualized != null)
            Destroy(lastVisualized);

        lastVisualized = snapshot;
        resultImage.texture = lastVisualized;
        SyncResultAspect(lastVisualized);
    }

    List<Detection> GetDisplayDetections(Texture displayFrame, List<Detection> detections)
    {
        if (displayFrame == null || detections == null || detections.Count == 0)
            return detections;

        if (cameraMgr == null || cameraMgr.OutputTexture == null)
            return detections;

        if (displayFrame == cameraMgr.OutputTexture)
            return detections;

        int outputWidth = cameraMgr.OutputTexture.width;
        int outputHeight = cameraMgr.OutputTexture.height;

        var mapped = new List<Detection>(detections.Count);
        foreach (var det in detections)
        {
            Rect rect = det.GetRect();
            Rect mappedRect = cameraMgr.MapRectFromOutputToFrame(rect, outputWidth, outputHeight);
            mapped.Add(new Detection
            {
                x1 = mappedRect.x,
                y1 = mappedRect.y,
                x2 = mappedRect.x + mappedRect.width,
                y2 = mappedRect.y + mappedRect.height,
                confidence = det.confidence,
                classId = det.classId,
                className = det.className
            });
        }

        return mapped;
    }

    void SyncResultAspect(Texture texture)
    {
        if (texture == null || resultImage == null)
            return;

        var rect = resultImage.rectTransform;
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;

        var fitter = resultImage.GetComponent<AspectRatioFitter>();
        if (fitter != null)
        {
            float aspect = (float)texture.width / Mathf.Max(1, texture.height);
            fitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
            fitter.aspectRatio = aspect;
        }
    }

    Texture2D CopyToTexture2D(Texture2D source)
    {
        RenderTexture rt = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(source, rt);
        RenderTexture active = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D copy = new Texture2D(source.width, source.height, TextureFormat.RGBA32, false);
        copy.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
        copy.Apply();

        RenderTexture.active = active;
        RenderTexture.ReleaseTemporary(rt);
        return copy;
    }

    void DrawDetections(Texture2D texture, List<Detection> detections)
    {
        if (texture == null || detections == null)
            return;

        int width = texture.width;
        int height = texture.height;

        foreach (var det in detections)
        {
            Color color = classColors.Length > 0 ? classColors[det.classId % classColors.Length] : Color.red;

            int x1 = Mathf.Clamp(Mathf.RoundToInt(det.x1), 0, width - 1);
            int y1 = Mathf.Clamp(Mathf.RoundToInt(det.y1), 0, height - 1);
            int x2 = Mathf.Clamp(Mathf.RoundToInt(det.x2), 0, width - 1);
            int y2 = Mathf.Clamp(Mathf.RoundToInt(det.y2), 0, height - 1);

            // Texture2D 坐标原点在左下，需要做 Y 翻转
            int y1Tex = Mathf.Clamp(height - 1 - y2, 0, height - 1);
            int y2Tex = Mathf.Clamp(height - 1 - y1, 0, height - 1);

            DrawRect(texture, x1, y1Tex, x2, y2Tex, color, boxLineWidth);
        }

        texture.Apply();
    }

    void DrawRect(Texture2D tex, int x1, int y1, int x2, int y2, Color color, int lineWidth)
    {
        int minX = Mathf.Min(x1, x2);
        int maxX = Mathf.Max(x1, x2);
        int minY = Mathf.Min(y1, y2);
        int maxY = Mathf.Max(y1, y2);

        for (int i = 0; i < lineWidth; i++)
        {
            int top = Mathf.Clamp(maxY - i, 0, tex.height - 1);
            int bottom = Mathf.Clamp(minY + i, 0, tex.height - 1);
            for (int x = minX; x <= maxX; x++)
            {
                if (top >= 0 && top < tex.height) tex.SetPixel(x, top, color);
                if (bottom >= 0 && bottom < tex.height) tex.SetPixel(x, bottom, color);
            }

            int left = Mathf.Clamp(minX + i, 0, tex.width - 1);
            int right = Mathf.Clamp(maxX - i, 0, tex.width - 1);
            for (int y = minY; y <= maxY; y++)
            {
                if (left >= 0 && left < tex.width) tex.SetPixel(left, y, color);
                if (right >= 0 && right < tex.width) tex.SetPixel(right, y, color);
            }
        }
    }

    /// <summary>
    /// 在场景中绘制检测框
    /// </summary>
    void OnGUI()
    {
        if (!drawGizmos || currentDetections == null || currentDetections.Count == 0)
            return;

        // 使用 RawImage 绘制结果时，避免重复叠加 GUI 框
        if (updateResultImage && resultImage != null)
            return;

        if (labelStyle == null || lineTexture == null)
            InitializeGuiResources();

        var guiDetections = GetDisplayDetections(resultImage != null ? resultImage.texture as Texture2D : null, currentDetections);
        foreach (var det in guiDetections)
        {
            Color color = classColors[det.classId % classColors.Length];
            Rect rect = det.GetRect();

            if (rect.width <= 1f || rect.height <= 1f)
                continue;

            DrawRectOutline(rect, color, boxLineWidth);

            string label = $"{det.className} {det.confidence:F2}";
            GUI.color = Color.white;
            GUI.Label(new Rect(rect.x, Mathf.Max(0, rect.y - guiFontSize - 2), 260, guiFontSize + 4), label, labelStyle);
        }

        GUI.color = Color.white;
    }

    void DrawRectOutline(Rect rect, Color color, int lineWidth)
    {
        GUI.color = color;
        for (int i = 0; i < lineWidth; i++)
        {
            GUI.DrawTexture(new Rect(rect.x, rect.y + i, rect.width, 1), lineTexture);
            GUI.DrawTexture(new Rect(rect.x, rect.y + rect.height - i - 1, rect.width, 1), lineTexture);
            GUI.DrawTexture(new Rect(rect.x + i, rect.y, 1, rect.height), lineTexture);
            GUI.DrawTexture(new Rect(rect.x + rect.width - i - 1, rect.y, 1, rect.height), lineTexture);
        }
    }

    void OnDestroy()
    {
        if (detectRoutine != null)
        {
            StopCoroutine(detectRoutine);
            detectRoutine = null;
        }

        if (cameraMgr != null)
        {
            cameraMgr.StopCamera();
        }

        if (lastVisualized != null)
        {
            Destroy(lastVisualized);
        }
    }
}
