import cv2
from ultralytics import YOLO
import mediapipe as mp

# --- 画像ファイル名（掃除中の静止画）
IMAGE_PATH = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\cleaning.jpg"

# -------------------------
# YOLOv8 で物体検出
# -------------------------
yolo_model = YOLO("yolov8s.pt")  # 軽量モデル（必要なら yolov8s.pt にしてもOK）
yolo_results = yolo_model(IMAGE_PATH)

detected_objects = set()
for r in yolo_results:
    for box in r.boxes:
        cls_id = int(box.cls)
        label = yolo_model.names[cls_id]
        detected_objects.add(label)

print(f"🧹 検出された物体: {detected_objects}")

# -------------------------
# MediaPipe で姿勢検出
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

# 簡易的に「前傾 or しゃがみ」を判定（腰が低ければしゃがみっぽい）
is_bending = False
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    # Yの差で前傾姿勢を判定（簡易だけど効果大）
    if nose.y - left_hip.y < -0.05:
        is_bending = True

print(f"🧍‍♀️ 姿勢：{'前傾姿勢' if is_bending else '立ち or 中立'}")

# -------------------------
# 要約処理（シンプルルール）
# -------------------------
if "vacuum cleaner" in detected_objects and is_bending:
    print("📝 この人は『掃除機をかけている』と推定されます。")
elif "vacuum cleaner" in detected_objects:
    print("📝 掃除機を持っているけど、掃除中かは不明です。")
else:
    print("📝 掃除中ではない可能性があります。")
