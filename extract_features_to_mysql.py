import os
import pandas as pd
from audio_features import extract_audio_features
from sklearn.preprocessing import StandardScaler
import pymysql
from sqlalchemy import create_engine, text

DB_DIR = "audio_db"
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_HOST = 'localhost'
MYSQL_PORT = '3306'
MYSQL_DB = 'audio_features_db'
TABLE_NAME = 'features'

feature_cols = ['energy','zero_crossing_rate','percent_silence','bandwidth','spectral_centroid','harmonicity','pitch']

def get_label_from_filename(filename):
    fn = filename.lower()
    if "flu" in fn: return "Flute"
    if "cla" in fn: return "Clarinet"
    if "sax" in fn: return "Saxophone"
    if "tru" in fn: return "Trumpet"
    if "har" in fn: return "Harmonica"
    return "Unknown"

# 1. Kết nối tới MySQL để tạo DB và bảng nếu chưa có
conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, port=int(MYSQL_PORT), charset='utf8mb4', autocommit=True)
cur = conn.cursor()
cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
conn.select_db(MYSQL_DB)
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        label VARCHAR(50),
        energy DOUBLE,
        zero_crossing_rate DOUBLE,
        percent_silence DOUBLE,
        bandwidth DOUBLE,
        spectral_centroid DOUBLE,
        harmonicity DOUBLE,
        pitch DOUBLE,
        energy_scaled DOUBLE,
        zero_crossing_rate_scaled DOUBLE,
        percent_silence_scaled DOUBLE,
        bandwidth_scaled DOUBLE,
        spectral_centroid_scaled DOUBLE,
        harmonicity_scaled DOUBLE,
        pitch_scaled DOUBLE
    );
""")
cur.close()
conn.close()

# 2. Xóa dữ liệu cũ trong bảng (mỗi lần chạy lại làm mới)
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4")
with engine.connect() as connection:
    connection.execute(text(f"DELETE FROM {TABLE_NAME}"))

# 3. Trích xuất đặc trưng
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

# 4. Chuẩn hóa đặc trưng
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

scaled_cols = [col + '_scaled' for col in feature_cols]
df_scaled_rename = df_scaled[feature_cols].copy()
df_scaled_rename.columns = scaled_cols

# 5. Gộp gốc và chuẩn hóa
df_mysql = pd.concat([df[['filename','label'] + feature_cols], df_scaled_rename], axis=1)

# 6. Ghi dữ liệu mới vào bảng
df_mysql.to_sql(TABLE_NAME, engine, if_exists='append', index=False)

print(f"Đã làm mới dữ liệu và lưu đặc trưng gốc + chuẩn hóa vào bảng {TABLE_NAME} (MySQL) thành công!")
