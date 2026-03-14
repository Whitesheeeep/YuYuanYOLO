using UnityEngine;
using System.Collections;

/// <summary>
/// YuYuan 检测客户端使用示例
/// 演示如何使用客户端与服务器通信
/// </summary>
public class YuYuanClientExample : MonoBehaviour
{
    [Header("客户端")]
    public YuYuanDetectionClient client;

    [Header("测试图像")]
    public Texture2D testImage;

    [Header("摄像头")]
    public bool useWebcam = false;
    private WebCamTexture webcamTexture;

    [Header("可视化")]
    public bool drawDetections = true;
    public Color[] classColors = new Color[]
    {
        Color.red,      // Stone1
        Color.cyan,     // Picture
        Color.blue,     // LionLeft
        Color.yellow,   // LionRight
        Color.green     // Stone2
    };

    private DetectionResponse currentResponse;
    private Texture2D currentFrame;

    void Start()
    {
        if (client == null)
        {
            client = GetComponent<YuYuanDetectionClient>();
        }

        // 订阅事件
        client.OnDetectionComplete += OnDetectionComplete;
        client.OnDetectionError += OnDetectionError;

        // 检查服务器健康状态
        client.CheckServerHealth();

        // 启动摄像头
        if (useWebcam)
        {
            StartWebcam();
        }
    }

    void Update()
    {
        // 按空格键检测
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (useWebcam && webcamTexture != null && webcamTexture.isPlaying)
            {
                DetectFromWebcam();
            }
            else if (testImage != null)
            {
                DetectFromImage();
            }
        }

        // 按 H 键检查服务器健康
        if (Input.GetKeyDown(KeyCode.H))
        {
            client.CheckServerHealth();
        }
    }

    /// <summary>
    /// 从测试图像检测
    /// </summary>
    public void DetectFromImage()
    {
        if (testImage == null)
        {
            Debug.LogWarning("测试图像未设置！");
            return;
        }

        Debug.Log("发送图像到服务器...");
        client.DetectImage(testImage);
    }

    /// <summary>
    /// 启动摄像头
    /// </summary>
    void StartWebcam()
    {
        webcamTexture = new WebCamTexture(640, 480, 30);
        webcamTexture.Play();
        currentFrame = new Texture2D(640, 480);
        Debug.Log("摄像头已启动");
    }

    /// <summary>
    /// 从摄像头检测
    /// </summary>
    void DetectFromWebcam()
    {
        if (webcamTexture == null || !webcamTexture.isPlaying)
        {
            Debug.LogWarning("摄像头未启动！");
            return;
        }

        // 捕获当前帧
        currentFrame.SetPixels(webcamTexture.GetPixels());
        currentFrame.Apply();

        Debug.Log("发送摄像头图像到服务器...");
        client.DetectImage(currentFrame);
    }

    /// <summary>
    /// 检测完成回调
    /// </summary>
    void OnDetectionComplete(DetectionResponse response)
    {
        currentResponse = response;

        Debug.Log($"检测完成: {response.count} 个目标");
        foreach (var det in response.detections)
        {
            Debug.Log($"  - {det}");
        }
    }

    /// <summary>
    /// 检测错误回调
    /// </summary>
    void OnDetectionError(string error)
    {
        Debug.LogError($"检测错误: {error}");
    }

    /// <summary>
    /// 绘制检测框
    /// </summary>
    void OnGUI()
    {
        if (!drawDetections || currentResponse == null || currentResponse.detections == null)
            return;

        // 绘制检测结果
        foreach (var det in currentResponse.detections)
        {
            // 设置颜色
            Color color = classColors[det.class_id % classColors.Length];
            GUI.color = color;

            // 绘制边界框
            Rect rect = det.GetRect();
            GUI.Box(rect, "");

            // 绘制标签
            string label = $"{det.class_name} {det.confidence:F2}";
            GUI.Label(new Rect(rect.x, rect.y - 20, 200, 20), label);
        }

        GUI.color = Color.white;

        // 显示统计信息
        GUI.Label(new Rect(10, 10, 300, 20), $"检测数量: {currentResponse.count}");
        GUI.Label(new Rect(10, 30, 300, 20), $"推理时间: {currentResponse.inference_time * 1000:F1}ms");
    }

    void OnDestroy()
    {
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }

        // 取消订阅
        if (client != null)
        {
            client.OnDetectionComplete -= OnDetectionComplete;
            client.OnDetectionError -= OnDetectionError;
        }
    }
}
