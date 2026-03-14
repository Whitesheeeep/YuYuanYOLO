using UnityEngine;

/// <summary>
/// 图像预处理工具类
/// 处理不同分辨率的输入图像
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>
    /// 调整图像大小并保持宽高比（添加黑边）
    /// </summary>
    public static Texture2D ResizeWithLetterbox(Texture2D source, int targetSize)
    {
        float scale;
        int offsetX;
        int offsetY;
        return ResizeWithLetterbox(source, targetSize, out scale, out offsetX, out offsetY);
    }

    /// <summary>
    /// 调整图像大小并保持宽高比（添加黑边），返回缩放与边距信息
    /// </summary>
    public static Texture2D ResizeWithLetterbox(Texture2D source, int targetSize, out float scale, out int offsetX, out int offsetY)
    {
        // 默认 padding 颜色对齐 Ultralytics (114)
        Color32 padColor = new Color32(114, 114, 114, 255);

        int sourceWidth = source.width;
        int sourceHeight = source.height;

        // 计算缩放比例
        scale = Mathf.Min(
            (float)targetSize / sourceWidth,
            (float)targetSize / sourceHeight
        );

        int newWidth = Mathf.RoundToInt(sourceWidth * scale);
        int newHeight = Mathf.RoundToInt(sourceHeight * scale);

        // 创建目标纹理（padding 背景）
        Texture2D result = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);
        Color[] pixels = new Color[targetSize * targetSize];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = padColor;
        }
        result.SetPixels(pixels);

        // 调整源图像大小
        RenderTexture rt = RenderTexture.GetTemporary(newWidth, newHeight);
        rt.filterMode = FilterMode.Bilinear;
        Graphics.Blit(source, rt);

        RenderTexture.active = rt;
        Texture2D resized = new Texture2D(newWidth, newHeight, TextureFormat.RGB24, false);
        resized.ReadPixels(new Rect(0, 0, newWidth, newHeight), 0, 0);
        resized.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        // 居中放置
        offsetX = (targetSize - newWidth) / 2;
        offsetY = (targetSize - newHeight) / 2;

        Color[] resizedPixels = resized.GetPixels();
        result.SetPixels(offsetX, offsetY, newWidth, newHeight, resizedPixels);
        result.Apply();

        return result;
    }

    /// <summary>
    /// 简单拉伸调整（不保持宽高比）
    /// </summary>
    public static Texture2D ResizeStretch(Texture2D source, int targetSize)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetSize, targetSize);
        rt.filterMode = FilterMode.Bilinear;

        RenderTexture.active = rt;
        Graphics.Blit(source, rt);

        Texture2D result = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, targetSize, targetSize), 0, 0);
        result.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        return result;
    }

    /// <summary>
    /// 裁剪中心区域（不保持宽高比）
    /// </summary>
    public static Texture2D ResizeCenterCrop(Texture2D source, int targetSize)
    {
        int sourceWidth = source.width;
        int sourceHeight = source.height;

        // 计算裁剪区域
        float scale = Mathf.Max(
            (float)targetSize / sourceWidth,
            (float)targetSize / sourceHeight
        );

        int scaledWidth = Mathf.RoundToInt(sourceWidth * scale);
        int scaledHeight = Mathf.RoundToInt(sourceHeight * scale);

        // 先缩放
        RenderTexture rt = RenderTexture.GetTemporary(scaledWidth, scaledHeight);
        rt.filterMode = FilterMode.Bilinear;
        Graphics.Blit(source, rt);

        // 裁剪中心
        int cropX = (scaledWidth - targetSize) / 2;
        int cropY = (scaledHeight - targetSize) / 2;

        RenderTexture.active = rt;
        Texture2D result = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(cropX, cropY, targetSize, targetSize), 0, 0);
        result.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        return result;
    }
}
