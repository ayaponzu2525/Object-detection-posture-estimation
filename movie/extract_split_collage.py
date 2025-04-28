import cv2
import os
import numpy as np
from ultralytics import YOLO

# 動画からフレームを抽出し、物体検出を行い、指定したパートのコラージュ画像を作成するスクリプト

# === 設定 ===
VIDEO_PATH = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\input_video.mp4"
OUTPUT_DIR = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\frames_3"
COLLAGE_OUTPUT_PATH = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\collage_part4.jpg"
YOLO_MODEL_PATH = "yolov8s.pt"  # 使用するYOLOモデル
TARGET_CLASS = "person"  # 検出対象クラス
FRAME_INTERVAL_SEC = 1  # 1秒ごとにフレームを取り出す
TARGET_PART = "part4"  # コラージュ対象パート
RESIZE_HEIGHT = 200  # コラージュ画像の高さ
MAX_WIDTH = 5000  # コラージュ画像の最大幅（ピクセル）

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
print("🎉 クロップ完了！")

# === コラージュ作成 ===
part_images = sorted([f for f in os.listdir(OUTPUT_DIR) if TARGET_PART in f and f.endswith(".jpg")])
image_list = []

for img_name in part_images:
    img_path = os.path.join(OUTPUT_DIR, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        h, w = img.shape[:2]
        resize_width = int(w * (RESIZE_HEIGHT / h))
        resized_img = cv2.resize(img, (resize_width, RESIZE_HEIGHT))
        image_list.append(resized_img)

if image_list:
    rows = []
    current_row = []
    current_width = 0

    for img in image_list:
        if current_width + img.shape[1] > MAX_WIDTH:
            row = np.hstack(current_row)
            rows.append(row)
            current_row = [img]
            current_width = img.shape[1]
        else:
            current_row.append(img)
            current_width += img.shape[1]

    if current_row:
        row = np.hstack(current_row)
        rows.append(row)

    collage = np.vstack(rows)
    cv2.imwrite(COLLAGE_OUTPUT_PATH, collage)
    print(f"🎉 コラージュ保存完了: {COLLAGE_OUTPUT_PATH}")
else:
    print("⚠ コラージュ対象画像が見つかりませんでした")
