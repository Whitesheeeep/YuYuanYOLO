# WebSocket 消息协议参考

## 1. 检测客户端 → 服务端

### 1.1 检测请求（旧协议 JSON）
```json
{
  "type": "detect",
  "device_id": "device_001",
  "device_name": "AR眼镜1",
  "image": "<base64>"
}
```

### 1.2 导览请求
```json
{
  "type": "guidance_request",
  "device_id": "device_001",
  "device_name": "AR眼镜1",
  "query": "这是什么建筑？",
  "image": "<base64>"
}
```

---

## 2. 服务端 → 检测客户端

### 2.1 检测结果
```json
{
  "type": "detection_result",
  "device_id": "device_001",
  "device_name": "AR眼镜1",
  "detections": [
    {"class": "三穗堂", "confidence": 0.92, "bbox": [100, 50, 300, 250]}
  ]
}
```

### 2.2 导览结果
```json
{
  "type": "guidance_result",
  "device_id": "device_001",
  "device_name": "AR眼镜1",
  "query": "这是什么建筑？",
  "answer": "这是三穗堂，始建于..."
}
```

### 2.3 命令（由 Monitor/控制客户端发起，服务端转发）
```json
{
  "type": "command",
  "button_id": 3,
  "button_name": "按钮3"
}
```

---

## 3. 控制客户端 → 服务端

### 3.1 注册为控制客户端
```json
{
  "type": "register_control",
  "client_id": "ctrl_001"
}
```

### 3.2 发送命令给指定检测客户端
```json
{
  "type": "send_command",
  "target_connection_id": "192.168.1.5:12345",
  "button_id": 3,
  "button_name": "按钮3"
}
```

### 3.3 清除指定客户端的对话历史
```json
{
  "type": "clear_history",
  "target_connection_id": "192.168.1.5:12345"
}
```
> `target_connection_id` 可省略，省略时清除发送者自身的会话历史。

---

## 4. 服务端 → 控制客户端

### 4.1 检测客户端列表（注册后立即下发，之后每次变动自动推送）
```json
{
  "type": "detection_client_list",
  "clients": [
    {
      "connection_id": "192.168.1.5:12345",
      "device_id": "device_001",
      "device_name": "AR眼镜1"
    }
  ]
}
```
