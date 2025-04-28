import cv2
import os
from ultralytics import YOLO


# 動画からフレームを抽出し、物体検出を行い、一番大きい人物の部分を４分割する

# === 設定 ===
VIDEO_PATH = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\drink_water.mp4"
OUTPUT_DIR = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\frames_3"
YOLO_MODEL_PATH = "yolov8s.pt"  # 使用するYOLOモデル
TARGET_CLASS = "person"  # 検出対象クラス
FRAME_INTERVAL_SEC = 1  # 1秒ごとにフレームを取り出す

# === 初期化 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(YOLO_MODEL_PATH)

# === 動画読み込み ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * FRAME_INTERVAL_SEC)

frame_count = 0
save_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # === YOLOで物体検出 ===
        results = model(frame)
        max_area = 0
        best_box = None

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                if label == TARGET_CLASS:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)

        if best_box:
            x1, y1, x2, y2 = map(int, best_box)
            person_crop = frame[y1:y2, x1:x2]

            # サイズ確認
            h, w = person_crop.shape[:2]
            if h >= 10 and w >= 10:
                # 縦2分割、横2分割
                half_h, half_w = h // 2, w // 2
                crops = [
                    person_crop[0:half_h, 0:half_w],      # 左上
                    person_crop[0:half_h, half_w:w],      # 右上
                    person_crop[half_h:h, 0:half_w],      # 左下
                    person_crop[half_h:h, half_w:w],      # 右下
                ]

                for idx, crop_img in enumerate(crops):
                    output_path = os.path.join(OUTPUT_DIR, f"frame{save_count:04d}_part{idx+1}.jpg")
                    cv2.imwrite(output_path, crop_img)

                print(f"✅ Saved frame{save_count:04d}")
                save_count += 1
            else:
                print(f"⚠ 小さすぎるcropをスキップ (frame {frame_count})")
        else:
            print(f"⚠ No person detected (frame {frame_count})")

    frame_count += 1

cap.release()
print("🎉 完了！")
