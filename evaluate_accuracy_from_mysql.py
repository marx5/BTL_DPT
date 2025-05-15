import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from audio_features import extract_audio_features

MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_HOST = 'localhost'
MYSQL_PORT = '3306'
MYSQL_DB = 'audio_features_db'
TABLE_NAME = 'features'
TEST_DIR = "audio_test"

feature_cols = ['energy','zero_crossing_rate','percent_silence','bandwidth','spectral_centroid','harmonicity','pitch']
scaled_cols = [c+'_scaled' for c in feature_cols]

def get_label_from_filename(filename):
    fn = filename.lower()
    if "flu" in fn: return "Flute"
    if "cla" in fn: return "Clarinet"
    if "sax" in fn: return "Saxophone"
    if "tru" in fn: return "Trumpet"
    if "har" in fn: return "Harmonica"
    return "Unknown"

def cosine_similarity(a, b):
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom if denom != 0 else 0

engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4")
db_df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

means = db_df[feature_cols].mean()
stds  = db_df[feature_cols].std()
X_db_scaled = db_df[scaled_cols].values
db_labels = db_df['label'].tolist()

num_test = 0
num_top1_correct = 0
results = []

print(f"Bắt đầu đánh giá với {TEST_DIR} ...")
for fname in os.listdir(TEST_DIR):
    if fname.lower().endswith(".wav"):
        fpath = os.path.join(TEST_DIR, fname)
        label_true = get_label_from_filename(fname)
        feats = extract_audio_features(fpath)
        q_vec_raw = np.array([feats[c] for c in feature_cols])
        q_vec_scaled = (q_vec_raw - means.values) / stds.values
        sim_scores = [cosine_similarity(q_vec_scaled, x) for x in X_db_scaled]
        idx_best = np.argmax(sim_scores)
        label_pred = db_labels[idx_best]
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

accuracy = num_top1_correct / num_test if num_test > 0 else 0
print(f"\nTổng số file test: {num_test}")
print(f"Số nhận dạng đúng: {num_top1_correct}")
print(f"Độ chính xác top-1: {accuracy*100:.2f}%")

pd.DataFrame(results).to_csv("test_accuracy_results.csv", index=False, encoding="utf-8-sig")
print("Đã lưu chi tiết vào test_accuracy_results.csv")
