import cv2
from ultralytics import YOLO
import math

# 学習済みモデル
model = YOLO("best260408.pt")

# パラメータ（CPU向け）
MOVE_THRESH = 4
STILL_FRAMES = 30
CONF = 0.25
IOU = 0.45
IMGSZ = 416


def snap_frame(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(video_path, "を開けません")
        return -1

    prev_center = {}
    still_count = 0
    frame_no = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # 2フレームに1回だけ推論（約2倍高速）
        if frame_no % 2 != 0:
            frame_no += 1
            continue

        results = model.track(
            frame,
            persist=True,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            verbose=False
        )

        moving = 0
        current_center = {}

        if len(results) > 0 and results[0].boxes is not None:

            boxes = results[0].boxes

            if boxes.id is not None:

                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()

                for track_id, box in zip(ids, xyxy):

                    x1, y1, x2, y2 = box

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    current_center[track_id] = (cx, cy)

                    if track_id in prev_center:

                        px, py = prev_center[track_id]

                        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                        if dist > MOVE_THRESH:
                            moving += 1

        if moving < 3:
            still_count += 1
        else:
            if still_count >= STILL_FRAMES:
                cap.release()
                return frame_no
            still_count = 0

        prev_center = current_center
        frame_no += 1

    cap.release()
    return -1


frame5 = snap_frame("ex5-26.mp4")

frame9 = snap_frame("ex9-26.mp4")   # ファイル名が ex9-26(1).mp4 の場合は変更


print("ex5 スナップフレーム =", frame5)
print("ex9 スナップフレーム =", frame9)

if frame5 != -1 and frame9 != -1:
    print("フレーム番号の差 =", abs(frame5 - frame9))
else:
    print("スナップフレームが見つかりませんでした。")