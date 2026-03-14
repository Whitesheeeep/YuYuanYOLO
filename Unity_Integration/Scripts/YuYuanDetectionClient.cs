using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// YuYuan 检测客户端
/// 通过 HTTP 与 Python 服务器通信
/// </summary>
public class YuYuanDetectionClient : MonoBehaviour
{
    [Header("服务器设置")]
    [Tooltip("检测服务器地址")]
    public string serverUrl = "http://localhost:5000";

    [Header("检测参数")]
    [Range(0.1f, 1f)]
    public float confidenceThreshold = 0.25f;

    [Header("超时设置")]
    public int timeoutSeconds = 30;

    // 检测结果回调
    public event Action<DetectionResponse> OnDetectionComplete;
    public event Action<string> OnDetectionError;

    /// <summary>
    /// 检测图像
    /// </summary>
    public void DetectImage(Texture2D image)
    {
        StartCoroutine(DetectImageCoroutine(image));
    }

    /// <summary>
    /// 检测图像协程
    /// </summary>
    private IEnumerator DetectImageCoroutine(Texture2D image)
    {
        // 1. 将图像编码为 Base64
        byte[] imageBytes = image.EncodeToPNG();
        string base64Image = Convert.ToBase64String(imageBytes);

        // 2. 构建请求数据
        DetectionRequest request = new DetectionRequest
        {
            image = base64Image,
            confidence = confidenceThreshold
        };

        string jsonData = JsonUtility.ToJson(request);

        // 3. 发送 HTTP POST 请求
        using (UnityWebRequest www = new UnityWebRequest($"{serverUrl}/detect", "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = timeoutSeconds;

            Debug.Log($"发送检测请求到: {serverUrl}/detect");

            // 发送请求
            yield return www.SendWebRequest();

            // 4. 处理响应
            if (www.result == UnityWebRequest.Result.Success)
            {
                string responseText = www.downloadHandler.text;
                Debug.Log($"收到响应: {responseText}");

                try
                {
                    DetectionResponse response = JsonUtility.FromJson<DetectionResponse>(responseText);

                    if (response.success)
                    {
                        Debug.Log($"检测成功: {response.count} 个目标, 用时 {response.inference_time * 1000:F1}ms");

                        // 触发回调
                        OnDetectionComplete?.Invoke(response);
                    }
                    else
                    {
                        Debug.LogError($"检测失败: {response.error}");
                        OnDetectionError?.Invoke(response.error);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"解析响应失败: {e.Message}");
                    OnDetectionError?.Invoke($"解析响应失败: {e.Message}");
                }
            }
            else
            {
                string error = $"请求失败: {www.error}";
                Debug.LogError(error);
                OnDetectionError?.Invoke(error);
            }
        }
    }

    /// <summary>
    /// 检查服务器健康状态
    /// </summary>
    public void CheckServerHealth()
    {
        StartCoroutine(CheckServerHealthCoroutine());
    }

    private IEnumerator CheckServerHealthCoroutine()
    {
        using (UnityWebRequest www = UnityWebRequest.Get($"{serverUrl}/health"))
        {
            www.timeout = 5;
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"服务器健康检查: {www.downloadHandler.text}");
            }
            else
            {
                Debug.LogError($"服务器连接失败: {www.error}");
            }
        }
    }
}

// ============================================================
// 数据结构
// ============================================================

/// <summary>
/// 检测请求
/// </summary>
[Serializable]
public class DetectionRequest
{
    public string image;        // Base64 编码的图像
    public float confidence;    // 置信度阈值
}

/// <summary>
/// 检测响应
/// </summary>
[Serializable]
public class DetectionResponse
{
    public bool success;
    public Detection[] detections;
    public int count;
    public float inference_time;
    public ImageSize image_size;
    public string error;
}

/// <summary>
/// 单个检测结果
/// </summary>
[Serializable]
public class Detection
{
    public int class_id;
    public string class_name;
    public float confidence;
    public BBox bbox;

    public Rect GetRect()
    {
        return new Rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
    }

    public override string ToString()
    {
        return $"{class_name} ({confidence:F2}) at ({bbox.x1:F0}, {bbox.y1:F0}, {bbox.x2:F0}, {bbox.y2:F0})";
    }
}

/// <summary>
/// 边界框
/// </summary>
[Serializable]
public class BBox
{
    public float x1, y1, x2, y2;
}

/// <summary>
/// 图像尺寸
/// </summary>
[Serializable]
public class ImageSize
{
    public int width;
    public int height;
}
