from ultralytics import YOLO

# YOLOv8n
model = YOLO("../yolov8n.pt", task="detect")

result = model(source="screen")

