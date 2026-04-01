from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/models/crop_weed_detection4/weights/best.pt")

# Image path
image_path = "test.jpg"

# Run detection
results = model(image_path)

# Show result
for r in results:
    img = r.plot()

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()