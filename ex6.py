import cv2
import math
from ultralytics import YOLO

# モデル読み込み
model = YOLO("best260408.pt")

# 動画読み込み
cap = cv2.VideoCapture("ex5-26.mp4")

# 動画情報
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 保存用動画
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("ex6_result.mp4", fourcc, fps, (width, height))

# 前フレームの中心座標を保存
previous_centers = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # トラッキング
    results = model.track(
        source=frame,
        persist=True,
        imgsz=1280,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    # 検出結果がない場合
    if result.boxes is None:
        out.write(frame)
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for box in result.boxes:

        cls = int(box.cls[0])
        name = result.names[cls]

        if name != "KG_helmet":
            continue

        # Track ID
        if box.id is None:
            continue

        track_id = int(box.id[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 中心座標
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 前フレームとの移動距離
        if track_id in previous_centers:

            px, py = previous_centers[track_id]

            distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

            # 4ピクセル以上移動：赤
            if distance >= 4:
                color = (0, 0, 255)
            # 4ピクセル未満：緑
            else:
                color = (0, 255, 0)

        else:
            # 初回検出は緑
            color = (0, 255, 0)

        # 赤枠・緑枠を描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID表示（任意）
        cv2.putText(
            frame,
            f"ID:{track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # 中心座標を更新
        previous_centers[track_id] = (cx, cy)

    # 保存・表示
    out.write(frame)
    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("処理終了")