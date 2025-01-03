from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov5s.pt")  # or "yolov8s.pt" for YOLOv8

# Train the model
model.train(
    data="data.yaml",      # Path to data.yaml
    epochs=80,             # Number of epochs
    imgsz=416,             # Image size
    batch=16,              # Batch size
    project="mask_detection",  # Project name
    name="yolov5_training" # Experiment name
)
