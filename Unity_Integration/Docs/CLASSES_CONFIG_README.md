# 类别名称配置说明

## 📄 classes.txt 文件格式

类别名称文件是一个简单的文本文件，每行一个类别名称。

### 格式示例

```
Stone1
Picture
LionLeft
LionRight
Stone2
```

### 规则

1. **每行一个类别**: 每个类别名称占一行
2. **顺序重要**: 类别顺序必须与训练时的顺序一致
3. **无空行**: 避免在类别之间添加空行
4. **UTF-8 编码**: 文件应使用 UTF-8 编码保存
5. **支持中文**: 可以使用中文类别名称

### 中文示例

```
石头1
图片
左侧狮子
右侧狮子
石头2
```

## 🎮 Unity 中使用

### 方法 1: 使用 Resources 文件夹（推荐）

1. 在 Unity 项目中创建 `Assets/Resources/` 文件夹
2. 将 `classes.txt` 放入 `Resources` 文件夹
3. 在 Inspector 中分配 TextAsset:
   - 选择 YuYuanDetector GameObject
   - 将 `classes.txt` 拖到 `Class Names File` 字段

### 方法 2: 使用 StreamingAssets 文件夹

如果需要在运行时动态加载或修改类别文件：

1. 创建 `Assets/StreamingAssets/` 文件夹
2. 将 `classes.txt` 放入此文件夹
3. 修改代码使用 `Application.streamingAssetsPath` 加载

```csharp
string path = Path.Combine(Application.streamingAssetsPath, "classes.txt");
string[] lines = File.ReadAllLines(path);
```

## 🔧 修改类别名称

### 步骤

1. 打开 `classes.txt` 文件
2. 修改类别名称（保持顺序不变）
3. 保存文件
4. 在 Unity 中重新加载场景或重启游戏

### 示例：改为中文

**修改前:**
```
Stone1
Picture
LionLeft
LionRight
Stone2
```

**修改后:**
```
石头1
图片
左侧狮子
右侧狮子
石头2
```

## ⚠️ 注意事项

### 1. 类别数量必须匹配

类别文件中的类别数量必须与模型训练时的类别数量一致。

- YuYuan 模型: **5 个类别**
- 如果数量不匹配，检测结果可能不正确

### 2. 类别顺序必须一致

类别的顺序必须与训练时的顺序完全一致：

```
ID 0: Stone1
ID 1: Picture
ID 2: LionLeft
ID 3: LionRight
ID 4: Stone2
```

如果顺序错误，检测到的类别名称会不正确。

### 3. 编码问题

- 使用 UTF-8 编码保存文件
- 避免使用 BOM (Byte Order Mark)
- Windows 记事本可能会添加 BOM，建议使用 VS Code 或 Notepad++

### 4. 换行符

- Windows: `\r\n` (CRLF)
- Unix/Mac: `\n` (LF)
- 代码会自动处理两种格式

## 🧪 测试类别加载

在 Unity Console 中查看日志：

```
从文件加载了 5 个类别:
  [0] Stone1
  [1] Picture
  [2] LionLeft
  [3] LionRight
  [4] Stone2
```

如果看到此日志，说明类别加载成功。

## 📝 代码示例

### 获取类别名称

```csharp
YuYuanDetector detector = GetComponent<YuYuanDetector>();

// 检测
List<Detection> detections = detector.Detect(image);

foreach (var det in detections)
{
    // 类别名称已自动从文件加载
    Debug.Log($"检测到: {det.className}");
}
```

### 动态修改类别名称

如果需要在运行时修改类别名称：

```csharp
// 创建新的 TextAsset
TextAsset newClassNames = Resources.Load<TextAsset>("new_classes");
detector.classNamesFile = newClassNames;

// 重新初始化
detector.InitializeModel();
```

## 🌍 多语言支持

### 创建多个类别文件

```
Assets/Resources/
├── classes_en.txt  (英文)
├── classes_zh.txt  (中文)
└── classes_ja.txt  (日文)
```

### 运行时切换语言

```csharp
public class LanguageManager : MonoBehaviour
{
    public YuYuanDetector detector;

    public void SetLanguage(string lang)
    {
        TextAsset classFile = Resources.Load<TextAsset>($"classes_{lang}");
        if (classFile != null)
        {
            detector.classNamesFile = classFile;
            detector.InitializeModel();
        }
    }
}
```

## ✅ 完成

现在你可以轻松修改类别名称，无需重新编译代码！
