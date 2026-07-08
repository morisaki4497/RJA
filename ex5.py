import cv2
from ultralytics import YOLO

# モデルの読み込み
model = YOLO("best260408.pt")

# 動画の読み込み
cap = cv2.VideoCapture("ex5-26.mp4")

# 動画情報
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 保存用動画
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("ex5_result.mp4", fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLOで検出
    results = model.predict(
        source=frame,
        imgsz=1280,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    count = 0

    # 検出結果を描画
    for box in results[0].boxes:

        cls = int(box.cls[0])
        name = results[0].names[cls]

        if name == "KG_helmet":

            count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 赤枠
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ラベル
            cv2.putText(
                frame,
                name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    # 30フレームごとに表示
    if frame_count % 30 == 0:
        print(f"{frame_count}フレーム: 青いヘルメット数 = {count}")

    # 動画へ保存
    out.write(frame)

    # 表示
    cv2.imshow("Result", frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
cap.release()
out.release()
cv2.destroyAllWindows()

print("処理終了")