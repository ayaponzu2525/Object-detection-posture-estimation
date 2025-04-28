import cv2
import os
import csv
from ultralytics import YOLO
import mediapipe as mp


# 動画からフレームを抽出し、物体検出と姿勢推定を行い、結果をCSVに保存するスクリプト

# === 設定 ===
FRAME_DIR = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\frames_2"  # フレーム画像が入ったフォルダ
OUTPUT_CSV = "frame_analysis_full_pose_2.csv"
YOLO_MODEL_PATH = "yolov8s.pt"  # YOLOモデル

# === 初期化 ===
model = YOLO(YOLO_MODEL_PATH)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# === MediaPipeの関節名リストを取得 ===
landmark_names = [landmark.name for landmark in mp_pose.PoseLandmark]

# === CSVのヘッダーを作成 ===
header = ["frame"]
for name in landmark_names:
    header.extend([f"{name}_x", f"{name}_y"])
header.append("detected_objects")

# === 出力CSVファイルを作成 ===
with open(OUTPUT_CSV, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # === フレーム画像の読み込み ===
    frame_files = sorted([f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")])
    for file in frame_files:
        frame_path = os.path.join(FRAME_DIR, file)
        image = cv2.imread(frame_path)

        # === YOLOで物体検出 ===
        results = model(image)
        objects = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                objects.add(label)

        # === MediaPipeで姿勢検出 ===
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        # === 33関節のx, y情報を格納 ===
        landmark_data = []
        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                landmark_data.append(round(lm.x, 5))
                landmark_data.append(round(lm.y, 5))
        else:
            landmark_data = [-1.0] * (len(landmark_names) * 2)  # 検出できなかったら-1

        # === 行を書き出す ===
        writer.writerow([file] + landmark_data + [";".join(objects)])
        print(f"✅ Processed {file}")
