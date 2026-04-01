from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=512,
    batch=8,
    project="models",
    name="crop_weed_detection"
)

print("Training Completed!")