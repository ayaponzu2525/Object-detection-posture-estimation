import cv2
import mediapipe as mp
import pandas as pd

# 動画ファイルのパス
VIDEO_PATH = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\cleaning_movie\self_movie\walk.mp4"
CSV_OUTPUT = "walk_pose_data.csv"

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# 動画読み込み
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
all_landmarks = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # BGR → RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = []
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        all_landmarks.append([frame_count] + landmarks)

cap.release()

# CSV列名を生成
columns = ['frame']
for i in range(33):
    columns += [f'{i}_x', f'{i}_y', f'{i}_z', f'{i}_vis']

# データフレーム化して保存
df = pd.DataFrame(all_landmarks, columns=columns)
df.to_csv(CSV_OUTPUT, index=False)
print(f"保存完了: {CSV_OUTPUT}")
