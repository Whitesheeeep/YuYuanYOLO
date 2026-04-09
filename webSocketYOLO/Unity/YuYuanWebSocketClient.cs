using UnityEngine;
using NativeWebSocket;
using System;
using System.Text;
using System.Collections;

namespace YOLO.WebSocket
{
    public class YuYuanWebSocketClient : MonoBehaviour
    {
        [Header("服务器设置")]
        public string serverUrl = "ws://192.168.1.100:5000";

        [Header("设备信息")]
        public string deviceId = "device_001";
        public string deviceName = "Android Device 1";

        [Header("摄像头设置")]
        public int targetWidth = 640;
        public int targetHeight = 480;
        public int targetFPS = 15;

        [Header("图像质量")]
        [Range(1, 100)]
        public int jpegQuality = 80;

        private NativeWebSocket.WebSocket websocket;
        /// <summary>暴露 WebSocket 实例，供 YuYuanCommandReceiver 等外部模块注册处理方法。</summary>
        public NativeWebSocket.WebSocket Websocket => websocket;
        private WebCamTexture webcamTexture;
        private Texture2D captureTexture;
        private bool isCapturing = false;

        async void Start()
        {
            // 创建 WebSocket 连接
            websocket = new NativeWebSocket.WebSocket(serverUrl);

            websocket.OnOpen += () =>
            {
                Debug.Log("WebSocket 已连接");
                StartCamera();
                StartCoroutine(CaptureLoop());
            };

            websocket.OnMessage += (bytes) =>
            {
                string message = Encoding.UTF8.GetString(bytes);
                HandleMessage(message);
            };

            websocket.OnError += (error) =>
            {
                Debug.LogError($"WebSocket 错误: {error}");
            };

            websocket.OnClose += (code) =>
            {
                Debug.Log($"WebSocket 已断开: {code}");
                isCapturing = false;
            };

            // 连接
            await websocket.Connect();
        }

        void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            websocket?.DispatchMessageQueue();
#endif
        }

        void StartCamera()
        {
            WebCamDevice[] devices = WebCamTexture.devices;

            if (devices.Length == 0)
            {
                Debug.LogError("未找到摄像头");
                return;
            }

            // 优先使用后置摄像头
            string deviceName = devices[0].name;
            foreach (var device in devices)
            {
                if (!device.isFrontFacing)
                {
                    deviceName = device.name;
                    break;
                }
            }

            webcamTexture = new WebCamTexture(deviceName, targetWidth, targetHeight, targetFPS);
            webcamTexture.Play();

            captureTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);

            Debug.Log($"摄像头已启动: {deviceName}");
        }

        IEnumerator CaptureLoop()
        {
            yield return new WaitForSeconds(1f);
            isCapturing = true;

            while (isCapturing)
            {
                if (webcamTexture != null && webcamTexture.isPlaying && websocket.State == WebSocketState.Open)
                {
                    CaptureAndSend();
                }

                yield return new WaitForSeconds(1f / targetFPS);
            }
        }

        async void CaptureAndSend()
        {
            // 捕获图像
            captureTexture.SetPixels(webcamTexture.GetPixels());
            captureTexture.Apply();

            // 编码为 JPEG
            byte[] imageData = captureTexture.EncodeToJPG(jpegQuality);
            string base64Image = Convert.ToBase64String(imageData);

            // 构建消息
            var message = new DetectionRequest
            {
                type = "detect",
                device_id = deviceId,
                device_name = deviceName,
                image = base64Image
            };

            string json = JsonUtility.ToJson(message);

            // 发送
            await websocket.SendText(json);
        }

        void HandleMessage(string message)
        {
            // 接收检测结果（可选：在 Unity 中显示）
            var result = JsonUtility.FromJson<DetectionResult>(message);

            if (result.type == "detection_result" && result.device_id == deviceId)
            {
                Debug.Log($"检测到 {result.detections.Length} 个目标");
            }
        }

        async void OnApplicationQuit()
        {
            isCapturing = false;

            if (webcamTexture != null)
            {
                webcamTexture.Stop();
            }

            if (websocket != null)
            {
                await websocket.Close();
            }
        }
    }

    [Serializable]
    public class DetectionRequest
    {
        public string type;
        public string device_id;
        public string device_name;
        public string image;
    }

    [Serializable]
    public class DetectionResult
    {
        public string type;
        public string device_id;
        public string device_name;
        public string original_image;
        public string detected_image;
        public Detection[] detections;
    }

    [Serializable]
    public class Detection
    {
        public string @class;
        public float confidence;
        public int[] bbox;
    }
}

