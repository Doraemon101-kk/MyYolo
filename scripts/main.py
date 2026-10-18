from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "runs/detect/penguin_bear_yolo11n_960/weights/best.pt"
)
IMAGE_PATH = (
    ROOT
    / "datasets/penguin_bear/images/val/robot_000002.jpg"
)


model = YOLO(MODEL_PATH)

result = model.predict(
    source=str(IMAGE_PATH),
    imgsz=960,
    conf=0.4,
    device=0,
    verbose=False,
)[0]

image = cv2.imread(str(IMAGE_PATH))

for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    class_name = result.names[class_id]

    print(
        f"类别：{class_name}，"
        f"置信度：{confidence:.2f}，"
        f"检测框：({x1}, {y1}) 到 ({x2}, {y2})"
    )
