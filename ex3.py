import cv2
from ultralytics import YOLO

# 学習済みモデル
model = YOLO("best260408.pt")

# 画像を読み込み
img = cv2.imread("ex1-26.png")

# 物体検出
results = model.predict(img, conf=0.1)

# 元画像
img = results[0].orig_img

# 検出結果
boxes = results[0].boxes

helmet_count = 0

for box in boxes:

    # 座標
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # クラスID
    cls = int(box.cls[0])

    # 信頼度
    conf = float(box.conf[0])

    # best260408.ptが青ヘルメットのみ学習している場合
    if conf >= 0.58:

        helmet_count += 1

        # 赤枠を描画
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        # 信頼度表示
        cv2.putText(
            img,
            f"{conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

# 検出数表示
cv2.putText(
    img,
    f"Blue Helmet : {helmet_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

print("青いヘルメット数 =", helmet_count)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()