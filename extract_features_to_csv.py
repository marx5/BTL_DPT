import os
import pandas as pd
from audio_features import extract_audio_features
from sklearn.preprocessing import StandardScaler
import joblib

DB_DIR = "audio_db"
OUTPUT_CSV = "audio_features_db.csv"
SCALER_PATH = "scaler_save.pkl"

def get_label_from_filename(filename):
    fn = filename.lower()
    if "flu" in fn: return "Flute"
    if "cla" in fn: return "Clarinet"
    if "sax" in fn: return "Saxophone"
    if "tru" in fn: return "Trumpet"
    if "har" in fn: return "Harmonica"
    return "Unknown"

if not os.path.isdir(DB_DIR):
    print(f"Thư mục {DB_DIR} không tồn tại!")
    exit(1)

features_list = []
print("Bắt đầu trích xuất đặc trưng...")

for fname in os.listdir(DB_DIR):
    if fname.lower().endswith(".wav"):
        fpath = os.path.join(DB_DIR, fname)
        try:
            feats = extract_audio_features(fpath)
            feats["filename"] = fname
            feats["label"] = get_label_from_filename(fname)
            features_list.append(feats)
            print(f"Đã trích xuất: {fname}")
        except Exception as e:
            print(f"Lỗi với file {fname}: {e}")

if not features_list:
    print("Không tìm thấy file .wav hợp lệ trong thư mục.")
    exit(2)

df = pd.DataFrame(features_list)
cols = ["filename", "label"] + [col for col in df.columns if col not in ["filename", "label"]]
df = df[cols]
# CHUẨN HÓA đặc trưng, trừ filename và label
feature_cols = [c for c in df.columns if c not in ["filename", "label"]]
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
df_scaled.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
joblib.dump(scaler, SCALER_PATH)
print(f"\nĐã lưu đặc trưng CHUẨN HÓA ra file {OUTPUT_CSV} và scaler vào {SCALER_PATH}")
