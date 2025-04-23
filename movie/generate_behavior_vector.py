import csv
import math

# --- 入出力ファイルパス ---
INPUT_CSV = r"C:\Users\ayapo\Documents\hightech_local\Object-detection-posture-estimation\movie\frame_analysis_full_pose.csv"
OUTPUT_CSV = "behavior_vector_full.csv"

# --- 距離を計算する関数 ---
def distance(x1, y1, x2, y2):
    if -1 in [x1, y1, x2, y2]:  # 欠損データの場合
        return -1
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# --- 入力CSVの読み込み ---
with open(INPUT_CSV, newline='') as f:
    reader = list(csv.DictReader(f))

# --- 出力CSVの書き出し ---
with open(OUTPUT_CSV, "w", newline='') as f:
    writer = csv.writer(f)

    # ヘッダー
    writer.writerow([
        "frame_pair",
        "left_hand_move", "right_hand_move",
        "left_foot_move", "right_foot_move",
        "left_hand_to_nose", "right_hand_to_nose",
        "hip_height_change",
        "object_change"
    ])

    # 各フレーム間を処理
    for i in range(len(reader) - 1):
        f1 = reader[i]
        f2 = reader[i + 1]

        # 座標取得関数
        def get_xy(frame, name):
            return float(frame.get(f"{name}_x", -1)), float(frame.get(f"{name}_y", -1))

        lh1, lh2 = get_xy(f1, "LEFT_WRIST"), get_xy(f2, "LEFT_WRIST")
        rh1, rh2 = get_xy(f1, "RIGHT_WRIST"), get_xy(f2, "RIGHT_WRIST")
        lf1, lf2 = get_xy(f1, "LEFT_ANKLE"), get_xy(f2, "LEFT_ANKLE")
        rf1, rf2 = get_xy(f1, "RIGHT_ANKLE"), get_xy(f2, "RIGHT_ANKLE")
        nose2 = get_xy(f2, "NOSE")
        lht2 = get_xy(f2, "LEFT_WRIST")
        rht2 = get_xy(f2, "RIGHT_WRIST")
        lhip1 = get_xy(f1, "LEFT_HIP")
        lhip2 = get_xy(f2, "LEFT_HIP")

        # 各距離・変化量の計算
        left_hand_move = distance(*lh1, *lh2)
        right_hand_move = distance(*rh1, *rh2)
        left_foot_move = distance(*lf1, *lf2)
        right_foot_move = distance(*rf1, *rf2)
        left_hand_to_nose = distance(*lht2, *nose2)
        right_hand_to_nose = distance(*rht2, *nose2)
        hip_height_change = abs(lhip2[1] - lhip1[1]) if -1 not in [lhip1[1], lhip2[1]] else -1

        # 物体の変化（YOLO検出物ラベル）
        set1 = set(f1["detected_objects"].split(";"))
        set2 = set(f2["detected_objects"].split(";"))
        object_change = int(set1 != set2)

        # 書き出し
        writer.writerow([
            f'{f1["frame"]}→{f2["frame"]}',
            round(left_hand_move, 5),
            round(right_hand_move, 5),
            round(left_foot_move, 5),
            round(right_foot_move, 5),
            round(left_hand_to_nose, 5),
            round(right_hand_to_nose, 5),
            round(hip_height_change, 5),
            object_change
        ])
