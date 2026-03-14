using UnityEngine;

/// <summary>
/// Simple camera manager for feeding frames into YuYuanDetector.
/// Provides Texture2D frames and optional resize output.
/// </summary>
public class CameraMgr : MonoBehaviour
{
    [Header("Camera Settings")]
    // Requested capture width (actual may differ based on device support)
    public int requestedWidth = 1280;
    // Requested capture height (actual may differ based on device support)
    public int requestedHeight = 720;
    // Requested capture FPS (actual may differ based on device support)
    public int requestedFPS = 30;
    // Specific device name; leave empty to use the first available camera
    public string deviceName = "";
    // Auto-start camera on Start()
    public bool autoStart = true;

    [Header("Output Settings")]
    // If true, resizes camera feed to target size for OutputTexture
    public bool cropToTarget = false;
    // Output width when cropToTarget is enabled
    public int targetWidth = 1280;
    // Output height when cropToTarget is enabled
    public int targetHeight = 720;

    // Latest full-resolution camera frame
    public Texture2D FrameTexture { get; private set; }
    // Output frame used by detector (either resized or original)
    public Texture2D OutputTexture { get; private set; }

    // WebCamTexture instance for device capture
    WebCamTexture webcam;
    // Reused pixel buffer to avoid allocations each frame
    Color32[] pixelBuffer;

    void Start()
    {
        if (autoStart)
        {
            StartCamera();
        }
    }

    public void StartCamera()
    {
        if (webcam != null)
        {
            if (!webcam.isPlaying)
                webcam.Play();
            return;
        }

        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0)
        {
            Debug.LogWarning("No camera devices found.");
            return;
        }

        string selectedDevice = deviceName;
        if (string.IsNullOrEmpty(selectedDevice))
        {
            selectedDevice = devices[0].name;
        }

        // Initialize and start the camera stream
        webcam = new WebCamTexture(selectedDevice, requestedWidth, requestedHeight, requestedFPS);
        webcam.Play();
    }

    public void StopCamera()
    {
        if (webcam != null)
        {
            if (webcam.isPlaying)
                webcam.Stop();
            Destroy(webcam);
            webcam = null;
        }

        if (FrameTexture != null)
        {
            Destroy(FrameTexture);
            FrameTexture = null;
        }

        if (OutputTexture != null && OutputTexture != FrameTexture)
        {
            Destroy(OutputTexture);
            OutputTexture = null;
        }

        pixelBuffer = null;
    }

    void Update()
    {
        if (webcam == null || !webcam.isPlaying)
            return;

        if (!webcam.didUpdateThisFrame)
            return;

        // Pull the latest frame into a CPU texture
        EnsureFrameTexture(webcam.width, webcam.height);
        webcam.GetPixels32(pixelBuffer);
        FrameTexture.SetPixels32(pixelBuffer);
        FrameTexture.Apply();

        if (cropToTarget)
        {
            EnsureOutputTexture(targetWidth, targetHeight);
            BlitToTexture(webcam, OutputTexture);
        }
        else
        {
            OutputTexture = FrameTexture;
        }
    }

    void EnsureFrameTexture(int width, int height)
    {
        if (FrameTexture != null && FrameTexture.width == width && FrameTexture.height == height)
            return;

        if (FrameTexture != null)
            Destroy(FrameTexture);

        FrameTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        pixelBuffer = new Color32[width * height];
    }

    void EnsureOutputTexture(int width, int height)
    {
        if (OutputTexture != null && OutputTexture.width == width && OutputTexture.height == height)
            return;

        if (OutputTexture != null && OutputTexture != FrameTexture)
            Destroy(OutputTexture);

        OutputTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
    }

    void BlitToTexture(Texture source, Texture2D destination)
    {
        RenderTexture rt = RenderTexture.GetTemporary(destination.width, destination.height, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(source, rt);
        RenderTexture active = RenderTexture.active;
        RenderTexture.active = rt;

        // Read pixels back from GPU into CPU texture
        destination.ReadPixels(new Rect(0, 0, destination.width, destination.height), 0, 0);
        destination.Apply();

        RenderTexture.active = active;
        RenderTexture.ReleaseTemporary(rt);
    }

    /// <summary>
    /// Map a rect from OutputTexture space to FrameTexture space.
    /// </summary>
    public Rect MapRectFromOutputToFrame(Rect rect, int outputWidth, int outputHeight)
    {
        if (FrameTexture == null || outputWidth <= 0 || outputHeight <= 0)
            return rect;

        int frameWidth = FrameTexture.width;
        int frameHeight = FrameTexture.height;

        if (frameWidth == outputWidth && frameHeight == outputHeight)
            return rect;

        float scaleX = (float)frameWidth / outputWidth;
        float scaleY = (float)frameHeight / outputHeight;

        return new Rect(rect.x * scaleX, rect.y * scaleY, rect.width * scaleX, rect.height * scaleY);
    }

    void OnDestroy()
    {
        StopCamera();
    }
}

