import cv2
import os

# === 設定項目 ===
VIDEO_PATH = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\drink_water.mp4"  # 動画ファイルのパス
INTERVAL_SECONDS = 0.5         # 0.5秒 or 1.0秒ごとに抽出

# === 出力フォルダ ===
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# === 動画読み込み ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps * INTERVAL_SECONDS)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ 保存: {filename}")
        saved_count += 1
    frame_count += 1

cap.release()
print(f"\n🎉 {saved_count} 枚のフレームを抽出しました！")
