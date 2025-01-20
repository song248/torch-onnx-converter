import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("./assets/yolov8m.pt")


image_path = "test.jpg"
image = cv2.imread(image_path)

results = model(image)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

output_path = "test_result.jpg"
cv2.imwrite(output_path, image)
print(f"Save to result image to {output_path}")