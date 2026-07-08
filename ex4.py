import cv2
from ultralytics import YOLO

# モデル読み込み
model = YOLO("best260408.pt")

# 元画像
img = cv2.imread("ex2-26.png")

# ===== フィールド部分だけ切り出す =====
x1, y1 = 120, 180
x2, y2 = 1220, 710

field = img[y1:y2, x1:x2]

# 推論
results = model.predict(
    source=field,
    imgsz=1920,
    conf=0.001,
    iou=0.45,
    verbose=False
)

count = 0

# 赤枠を元画像へ描画
for box in results[0].boxes:
    cls = int(box.cls[0])
    name = results[0].names[cls]

    if name == "KG_helmet":
        count += 1

        bx1, by1, bx2, by2 = map(int, box.xyxy[0])

        # 切り出し画像→元画像の座標へ変換
        bx1 += x1
        bx2 += x1
        by1 += y1
        by2 += y1

        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

print("青いヘルメット数 =", count)

cv2.imwrite("ex4_result.png", img)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()