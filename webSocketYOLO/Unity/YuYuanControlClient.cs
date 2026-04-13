using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using NativeWebSocket;
using TMPro;

namespace YuYuan.Control
{
    /// <summary>
    /// YuYuan 控制客户端
    /// 功能：连接服务端、获取检测客户端列表、向目标发送命令按钮。
    /// Inspector 拖拽绑定：ipInputField、portInputField、connectButton、
    ///                     targetDropdown、commandButtons(10个)
    /// </summary>
    public class YuYuanControlClient : MonoBehaviour
    {
        // ----------------------------------------------------------------
        // Inspector 绑定
        // ----------------------------------------------------------------
        [Header("连接设置")]
        public TMP_InputField ipInputField;
        public TMP_InputField portInputField;
        public Button connectButton;

        [Header("控制端标识")]
        public string clientId = "unity_control_client";

        [Header("目标选择")]
        public TMP_Dropdown targetDropdown;

        [Header("命令按钮（10个，按顺序对应 button_id 1~10）")]
        public Button[] commandButtons = new Button[10];

        [Header("清除历史")]
        public Button clearHistoryButton;

        [Header("Chat 发送")]
        public TMP_InputField chatInputField;
        public Button sendButton;

        // ----------------------------------------------------------------
        // 私有状态
        // ----------------------------------------------------------------
        private WebSocket _websocket;
        private readonly List<string> _connectionIds = new List<string>(); // 与 dropdown 选项一一对应

        // ----------------------------------------------------------------
        // 生命周期
        // ----------------------------------------------------------------

        private void Start()
        {
            // 连接按钮
            connectButton.onClick.AddListener(Connect);

            // 清除历史按钮
            if (clearHistoryButton != null)
                clearHistoryButton.onClick.AddListener(SendClearHistory);

            if (sendButton != null)
                sendButton.onClick.AddListener(SendChatQuery);

            // 命令按钮 1~10
            for (int i = 0; i < commandButtons.Length; i++)
            {
                int buttonId = i + 1;
                if (commandButtons[i] != null)
                    commandButtons[i].onClick.AddListener(() => SendCommand(buttonId));
            }
        }

        private void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            _websocket?.DispatchMessageQueue();
#endif
        }

        private async void OnApplicationQuit()
        {
            connectButton.onClick.RemoveListener(Connect);
            if (clearHistoryButton != null)
                clearHistoryButton.onClick.RemoveListener(SendClearHistory);
            if (sendButton != null)
                sendButton.onClick.RemoveListener(SendChatQuery);
            for (int i = 0; i < commandButtons.Length; i++)
            {
                if (commandButtons[i] != null)
                    commandButtons[i].onClick.RemoveAllListeners();
            }

            if (_websocket != null)
                await _websocket.Close();
        }

        // ----------------------------------------------------------------
        // 连接
        // ----------------------------------------------------------------

        public async void Connect()
        {
            string ip   = ipInputField != null ? ipInputField.text.Trim() : "127.0.0.1";
            string port = portInputField != null ? portInputField.text.Trim() : "5000";
            string url  = $"ws://{ip}:{port}";

            // 关闭旧连接
            if (_websocket != null && _websocket.State == WebSocketState.Open)
                await _websocket.Close();

            _websocket = new WebSocket(url);
            _websocket.OnOpen    += OnOpen;
            _websocket.OnMessage += OnMessage;
            _websocket.OnError   += OnError;
            _websocket.OnClose   += OnClose;

            Debug.Log($"[ControlClient] 正在连接 {url}");
            await _websocket.Connect();
        }

        // ----------------------------------------------------------------
        // WebSocket 事件处理
        // ----------------------------------------------------------------

        private async void OnOpen()
        {
            Debug.Log("[ControlClient] 已连接，发送 register_control");
            string json = $"{{\"type\":\"register_control\",\"client_id\":\"{clientId}\"}}";
            await _websocket.SendText(json);
        }

        private void OnMessage(byte[] bytes)
        {
            string raw = Encoding.UTF8.GetString(bytes);
            try
            {
                var msg = JsonUtility.FromJson<ServerMessage>(raw);
                if (msg.type == "detection_client_list")
                {
                    var listMsg = JsonUtility.FromJson<DetectionClientListMessage>(raw);
                    UpdateDropdown(listMsg.clients);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[ControlClient] 消息解析失败: {e.Message}");
            }
        }

        private void OnError(string error)
        {
            Debug.LogError($"[ControlClient] 错误: {error}");
        }

        private void OnClose(WebSocketCloseCode code)
        {
            Debug.Log($"[ControlClient] 连接关闭: {code}");
        }

        // ----------------------------------------------------------------
        // 命令发送
        // ----------------------------------------------------------------

        private async void SendCommand(int buttonId)
        {
            if (_websocket == null || _websocket.State != WebSocketState.Open)
            {
                Debug.LogWarning("[ControlClient] 未连接，无法发送命令");
                return;
            }

            int idx = targetDropdown != null ? targetDropdown.value : -1;
            if (idx < 0 || idx >= _connectionIds.Count)
            {
                Debug.LogWarning("[ControlClient] 请先选择目标设备");
                return;
            }

            string targetId = _connectionIds[idx];
            string buttonName = commandButtons.Length >= buttonId && commandButtons[buttonId - 1] != null
                ? commandButtons[buttonId - 1].GetComponentInChildren<TMP_Text>()?.text ?? buttonId.ToString()
                : buttonId.ToString();

            var cmd = new SendCommandMessage
            {
                type = "send_command",
                target_connection_id = targetId,
                button_id = buttonId,
                button_name = buttonName,
            };
            await _websocket.SendText(JsonUtility.ToJson(cmd));
            Debug.Log($"[ControlClient] 发送命令 button_id={buttonId}({buttonName}) → {targetId}");
        }

        private async void SendClearHistory()
        {
            if (_websocket == null || _websocket.State != WebSocketState.Open)
            {
                Debug.LogWarning("[ControlClient] 未连接，无法发送清除历史");
                return;
            }

            int idx = targetDropdown != null ? targetDropdown.value : -1;
            if (idx < 0 || idx >= _connectionIds.Count)
            {
                Debug.LogWarning("[ControlClient] 请先选择目标设备");
                return;
            }

            string targetId = _connectionIds[idx];
            string json = $"{{\"type\":\"clear_history\",\"target_connection_id\":\"{targetId}\"}}";
            await _websocket.SendText(json);
            Debug.Log($"[ControlClient] 清除历史 → {targetId}");
        }

        private async void SendChatQuery()
        {
            if (_websocket == null || _websocket.State != WebSocketState.Open)
            {
                Debug.LogWarning("[ControlClient] 未连接，无法发送 Chat");
                return;
            }

            int idx = targetDropdown != null ? targetDropdown.value : -1;
            if (idx < 0 || idx >= _connectionIds.Count)
            {
                Debug.LogWarning("[ControlClient] 请先选择目标设备");
                return;
            }

            string queryText = chatInputField != null ? chatInputField.text.Trim() : string.Empty;
            if (string.IsNullOrEmpty(queryText))
            {
                Debug.LogWarning("[ControlClient] Chat 输入为空，取消发送");
                return;
            }

            string targetId = _connectionIds[idx];
            var msg = new SendQueryMessage
            {
                type = "query",
                target_connection_id = targetId,
                query = queryText
            };

            await _websocket.SendText(JsonUtility.ToJson(msg));
            Debug.Log($"[ControlClient] 发送 Chat → {targetId}: {queryText}");

            if (chatInputField != null)
                chatInputField.text = string.Empty;
        }

        // ----------------------------------------------------------------
        // Dropdown 更新
        // ----------------------------------------------------------------

        private void UpdateDropdown(DetectionClientInfo[] clients)
        {
            _connectionIds.Clear();
            if (targetDropdown == null) return;

            targetDropdown.ClearOptions();
            var options = new List<string>();
            foreach (var c in clients)
            {
                _connectionIds.Add(c.connection_id);
                options.Add($"{c.device_name} [{c.device_id}]");
            }
            targetDropdown.AddOptions(options);
            Debug.Log($"[ControlClient] 检测客户端列表更新，共 {clients.Length} 台");
        }

        // ----------------------------------------------------------------
        // 消息数据结构
        // ----------------------------------------------------------------

        [Serializable]
        private class ServerMessage
        {
            public string type;
        }

        [Serializable]
        private class DetectionClientListMessage
        {
            public string type;
            public DetectionClientInfo[] clients;
        }

        [Serializable]
        private class DetectionClientInfo
        {
            public string connection_id;
            public string device_id;
            public string device_name;
        }

        [Serializable]
        private class SendCommandMessage
        {
            public string type;
            public string target_connection_id;
            public int    button_id;
            public string button_name;
        }

        [Serializable]
        private class SendQueryMessage
        {
            public string type;
            public string target_connection_id;
            public string query;
        }
    }
}
