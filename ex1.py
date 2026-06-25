import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex1-26.png", conf=0.1)

img = results[0].orig_img
boxes = results[0].boxes

person_count = 0

for box in boxes:
    cls = int(box.cls[0])      # クラスID
    conf = float(box.conf[0])  # 信頼度

    # person かつ 信頼度20%以上
    if cls == 0 and conf >= 0.2:
        person_count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            img,
            f"Person {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

cv2.putText(
    img,
    f"Person Count: {person_count}",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255, 0, 0),
    3
)

print(f"人数: {person_count}")

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()