import cv2
from ultralytics import YOLO

# モデル読み込み
model = YOLO("yolov8x.pt")

# 画像を読み込み
results = model.predict("ex2-26.png", conf=0.04)

img = results[0].orig_img
boxes = results[0].boxes

person_count = 0

for box in boxes:

    # personクラスのみ
    cls = int(box.cls[0])
    if cls != 0:
        continue

    # 座標
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # 足元座標
    foot_x = (x1 + x2) // 2
    foot_y = y2

    # フィールド外（上下）
    if foot_y < 230 or foot_y > 670:
        continue

    # 台形の左右境界
    left = int(300 - (foot_y - 230) * 0.18)
    right = int(1120 + (foot_y - 230) * 0.25)

    

    # フィールド外なら除外
    if foot_x < left or foot_x > right:
        continue

    person_count += 1

    # 赤枠
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 人数表示
cv2.putText(
    img,
    f"Person : {person_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2,
)

print("人物数 =", person_count)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()