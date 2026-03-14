"""
NMS 功能演示脚本

功能：演示 NMS（非极大值抑制）如何过滤重叠的检测框

测试方法：
1. 创建一张测试图像
2. 模拟多个重叠的检测框
3. 应用 NMS 前后对比
"""

import cv2
import numpy as np

def apply_nms(boxes, scores, iou_threshold=0.5):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def main():
    # 创建测试图像
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 模拟重叠的检测框（故意创建重叠）
    boxes = [
        [100, 100, 300, 300],  # 框 1
        [120, 120, 320, 320],  # 框 2（与框 1 重叠）
        [110, 110, 310, 310],  # 框 3（与框 1、2 重叠）
        [400, 200, 600, 400],  # 框 4（独立）
        [420, 220, 620, 420],  # 框 5（与框 4 重叠）
    ]

    scores = [0.9, 0.85, 0.8, 0.95, 0.7]  # 置信度

    # 应用 NMS 前：绘制所有框
    img_before = img.copy()
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_before, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_before, f'Box {i+1}: {score:.2f}',
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 应用 NMS
    keep_indices = apply_nms(boxes, scores, iou_threshold=0.5)

    # 应用 NMS 后：只绘制保留的框
    img_after = img.copy()
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        score = scores[idx]
        cv2.rectangle(img_after, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_after, f'Box {idx+1}: {score:.2f}',
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 添加标题
    cv2.putText(img_before, 'Before NMS (All Boxes)', (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img_after, 'After NMS (Filtered)', (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 显示结果
    combined = np.hstack([img_before, img_after])
    cv2.imshow('NMS Demo', combined)

    print("=" * 60)
    print("NMS 演示")
    print("=" * 60)
    print(f"原始检测框数量: {len(boxes)}")
    print(f"NMS 后保留框数量: {len(keep_indices)}")
    print(f"保留的框索引: {[i+1 for i in keep_indices]}")
    print("=" * 60)
    print("按任意键关闭窗口...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
