using NativeWebSocket;
using UnityEngine;

namespace YuYuan.Control
{
    /// <summary>
    /// YuYuan 命令接收器（被控制端模块）
    /// 完全模块化，与 YuYuanWebSocketClient 无关。
    ///
    /// 使用方式：
    ///   1. 将本脚本挂到任意 GameObject。
    ///   2. 在持有 WebSocket 的脚本中调用 RegisterHandlers(ws) 完成接入。
    ///   3. 在 OnDestroy / 断开连接时调用 UnregisterHandlers(ws) 解除绑定。
    ///   4. 在下方各空方法内填充业务逻辑。
    /// </summary>
    public class YuYuanCommandReceiver : MonoBehaviour
    {
        // ----------------------------------------------------------------
        // 注册 / 注销
        // ----------------------------------------------------------------

        /// <summary>将本模块的处理方法注册到指定 WebSocket 实例。</summary>
        public void RegisterHandlers(WebSocket ws)
        {
            ws.OnOpen    += OnConnected;
            ws.OnMessage += OnMessageReceived;
            ws.OnError   += OnErrorOccurred;
            ws.OnClose   += OnDisconnected;
        }

        /// <summary>从指定 WebSocket 实例注销本模块的处理方法。</summary>
        public void UnregisterHandlers(WebSocket ws)
        {
            ws.OnOpen    -= OnConnected;
            ws.OnMessage -= OnMessageReceived;
            ws.OnError   -= OnErrorOccurred;
            ws.OnClose   -= OnDisconnected;
        }

        // ----------------------------------------------------------------
        // 处理方法（体全部留空，由使用者填充）
        // ----------------------------------------------------------------

        private void OnConnected()
        {
        }

        private void OnMessageReceived(byte[] bytes)
        {
        }

        private void OnCommandReceived(int buttonId, string buttonName)
        {
        }

        private void OnErrorOccurred(string error)
        {
        }

        private void OnDisconnected(WebSocketCloseCode code)
        {
        }
    }
}
