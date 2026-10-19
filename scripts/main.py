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

if image is None:
    raise FileNotFoundError(f"无法读取图片：{IMAGE_PATH}")

output_dir = ROOT / "runs/final_preview/rois"
output_dir.mkdir(parents=True, exist_ok=True)

mask_output_dir = ROOT / "runs/final_preview/masks"
mask_output_dir.mkdir(parents=True, exist_ok=True)

filled_output_dir = ROOT / "runs/final_preview/filled_masks"
filled_output_dir.mkdir(parents=True, exist_ok=True)

height, width = image.shape[:2]

for index, box in enumerate(result.boxes, start=1):
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    class_name = result.names[class_id]

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        print(f"跳过无效检测框：{class_name}")
        continue

    roi = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    #反向二值化
    _, mask = cv2.threshold(
        blurred,0,255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))

    mask = cv2.morphologyEx(mask,
                            cv2.MORPH_OPEN,
                            kernel,
                            iterations=1)

    mask = cv2.morphologyEx(mask,
                            cv2.MORPH_CLOSE,
                            kernel,
                            iterations=2)

    contours, _ = cv2.findContours(mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"没有找到轮廓：{class_name}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)

    contour_area = cv2.contourArea(largest_contour)

    moments = cv2.moments(largest_contour)

    if moments["m00"] == 0:
        print(f"轮廓面积为零，无法计算质心：{class_name}")
        continue

    local_center_x = int(moments["m10"] / moments["m00"])
    local_center_y = int(moments["m01"] / moments["m00"])

    global_center_x = x1 + local_center_x
    global_center_y = y1 + local_center_y
    print(f"{class_name} 质心："f"局部坐标 ({local_center_x}, {local_center_y})，"f"原图坐标 ({global_center_x}, {global_center_y})")


    filled_mask = mask.copy()
    filled_mask[:] = 0
    #把目标内部阴影异常部分也填满覆盖
    cv2.drawContours(
        filled_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    filled_output_path = (
        filled_output_dir
        / f"{index}_{class_name}_filled.png"
    )

    filled_saved = cv2.imwrite(
        str(filled_output_path),
        filled_mask)

    if filled_saved:
        print(
            f"填充轮廓已保存：{filled_output_path}，"
            f"轮廓面积：{contour_area:.0f} 像素"
        )
    else:
        print(f"填充轮廓保存失败：{filled_output_path}")
    mask_output_path = (mask_output_dir / f"{index}_{class_name}_mask.png")

    mask_saved = cv2.imwrite(str(mask_output_path),mask)   

    output_path = output_dir / f"{index}_{class_name}.jpg"
    saved = cv2.imwrite(str(output_path), roi)

    print(
        f"类别：{class_name}，"
        f"置信度：{confidence:.2f}，"
        f"检测框：({x1}, {y1}) 到 ({x2}, {y2})"
        )      

    if saved:
        print(f"目标区域已保存：{output_path}")
    else:
        print(f"目标区域保存失败：{output_path}")

    if mask_saved:
        print(f"黑白遮罩已保存：{mask_output_path}")
    else:
        print(f"黑白遮罩保存失败：{mask_output_path}")