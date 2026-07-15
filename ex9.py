import cv2
import math
from ultralytics import YOLO

# モデル読み込み
model = YOLO("best260408.pt")

# 動画読み込み
cap = cv2.VideoCapture("ex9-26.mp4")

# 動画情報
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 保存用動画
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("ex9_result.mp4", fourcc, fps, (width, height))

# 前フレームの中心座標
previous_centers = {}

# TEAM STATIONARYが続いたフレーム数
stationary_count = 0

# 保存したスナップ枚数
snap_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    moving_count = 0

    # トラッキング
    results = model.track(
        source=frame,
        persist=True,
        imgsz=1920,
        conf=0.10,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        for box in result.boxes:

            cls = int(box.cls[0])
            name = result.names[cls]

            if name != "KG_helmet":
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 中心座標
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (0, 255, 0)

            if track_id in previous_centers:

                px, py = previous_centers[track_id]

                distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                if distance >= 4:
                    color = (0, 0, 255)
                    moving_count += 1

            # 枠を描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ID表示
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # 中心座標更新
            previous_centers[track_id] = (cx, cy)

    # TEAM状態判定
    if moving_count >= 3:

        text = "TEAM MOVING"
        text_color = (0, 0, 255)

        # 30フレーム以上静止後に動いた瞬間を保存
        if stationary_count >= 30:

            snap_count += 1

            filename = f"snap_{snap_count}.png"

            cv2.imwrite(filename, frame)

            print(filename, "を保存しました")

        stationary_count = 0

    else:

        text = "TEAM STATIONARY"
        text_color = (0, 255, 0)

        stationary_count += 1

    # 右上へ表示
    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        2
    )[0]

    x = width - text_size[0] - 20
    y = 40

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        2
    )

    # 動画保存
    out.write(frame)

    # 表示
    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("処理終了")