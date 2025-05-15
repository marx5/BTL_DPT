import os
import pandas as pd
import numpy as np
import joblib
from audio_features import extract_audio_features

# Đường dẫn
DB_CSV = "audio_features_db.csv"
SCALER_PATH = "scaler_save.pkl"
DB_DIR = "audio_db"
TEST_DIR = "audio_test"

# Đặc trưng cần dùng
feature_cols = ['energy','zero_crossing_rate','percent_silence',
                'bandwidth','spectral_centroid','harmonicity','pitch']

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom if denom != 0 else 0

def get_label_from_filename(filename):
    fn = filename.lower()
    if "flu" in fn: return "Flute"
    if "cla" in fn: return "Clarinet"
    if "sax" in fn: return "Saxophone"
    if "tru" in fn: return "Trumpet"
    if "har" in fn: return "Harmonica"
    return "Unknown"

# Đọc cơ sở dữ liệu đặc trưng
db_df = pd.read_csv(DB_CSV)
scaler = joblib.load(SCALER_PATH)
X_db = db_df[feature_cols].values
db_labels = db_df['label'].tolist()

# Thống kê
num_test = 0
num_top1_correct = 0

# Ghi kết quả từng file nếu muốn
results = []

print(f"Bắt đầu đánh giá với {TEST_DIR} ...")
for fname in os.listdir(TEST_DIR):
    if fname.lower().endswith(".wav"):
        fpath = os.path.join(TEST_DIR, fname)
        # 1. Lấy nhãn thực từ tên file test
        label_true = get_label_from_filename(fname)
        # 2. Trích xuất đặc trưng, chuẩn hóa
        feats = extract_audio_features(fpath)
        q_vec_df = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)
        q_vec_scaled = scaler.transform(q_vec_df)[0]

        # 3. So sánh cosine với toàn bộ DB
        sim_scores = [cosine_similarity(q_vec_scaled, x) for x in X_db]
        # 4. Tìm index có cosine lớn nhất
        idx_best = np.argmax(sim_scores)
        label_pred = db_labels[idx_best]
        # 5. Ghi nhận kết quả
        num_test += 1
        correct = (label_pred == label_true)
        if correct: num_top1_correct += 1
        results.append({
            "file": fname,
            "true_label": label_true,
            "pred_label": label_pred,
            "top1_cosine": sim_scores[idx_best],
            "is_correct": correct
        })
        print(f"{fname}: True={label_true}, Pred={label_pred}, Cosine={sim_scores[idx_best]:.4f}, {'OK' if correct else 'WRONG'}")

# Tính độ chính xác
accuracy = num_top1_correct / num_test if num_test > 0 else 0
print(f"\nTổng số file test: {num_test}")
print(f"Số nhận dạng đúng: {num_top1_correct}")
print(f"Độ chính xác top-1: {accuracy*100:.2f}%")

# Lưu file kết quả chi tiết (nếu muốn)
pd.DataFrame(results).to_csv("test_accuracy_results.csv", index=False, encoding="utf-8-sig")
print("Đã lưu chi tiết vào test_accuracy_results.csv")
