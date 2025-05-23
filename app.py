import streamlit as st
import os
import pandas as pd
import numpy as np
import tempfile
from sqlalchemy import create_engine
from datetime import datetime
from audio_features import extract_audio_features

# Thông tin MySQL
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_HOST = 'localhost'
MYSQL_PORT = '3306'
MYSQL_DB = 'audio_features_db'
TABLE_NAME = 'features'

DB_DIR = "audio_db"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

feature_cols = ['energy','zero_crossing_rate','percent_silence','bandwidth','spectral_centroid','harmonicity','pitch']
scaled_cols = [c+'_scaled' for c in feature_cols]
FEATURE_NAMES = {
    "energy": "Năng lượng",
    "zero_crossing_rate": "Tốc độ đổi dấu",
    "percent_silence": "Khoảng lặng (%)",
    "bandwidth": "Băng thông (Hz)",
    "spectral_centroid": "Trung tâm phổ (Hz)",
    "harmonicity": "Độ điều hòa (dB)",
    "pitch": "Độ cao (Hz)"
}

@st.cache_data
def load_db_data():
    """
    Tải dữ liệu từ cơ sở dữ liệu và tính toán các giá trị trung bình, độ lệch chuẩn.
    Kết quả của hàm này sẽ được cache lại bởi Streamlit.
    """
    engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4")
    db_df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    means = db_df[feature_cols].mean()
    stds  = db_df[feature_cols].std()
    return db_df, means, stds

@st.cache_data
def get_audio_features(file_path, save_spectrogram_path=None):
    """
    Trích xuất đặc trưng âm thanh cho một file.
    Sử dụng cache để tránh tính toán lại nếu file và đường dẫn lưu không đổi.
    """
    return extract_audio_features(file_path, save_spectrogram_path=save_spectrogram_path)

def cosine_similarity_weighted(vec1, vec2, weights):
    num = np.sum(weights * vec1 * vec2)
    denom = np.sqrt(np.sum(weights * vec1 ** 2)) * np.sqrt(np.sum(weights * vec2 ** 2))
    return num / denom if denom != 0 else 0


def clear_outputs_folder(outputs_dir):
    import glob
    for file in glob.glob(os.path.join(outputs_dir, "*")):
        try:
            os.remove(file)
        except: pass

st.set_page_config(page_title="Truy vấn nhạc cụ bộ hơi", layout="wide")
st.title("🎷 Hệ thống truy vấn & nhận dạng nhạc cụ bộ hơi (MySQL)")

left, right = st.columns([1,2])

with left:
    uploaded_file = st.file_uploader("🎼 Chọn file .wav truy vấn", type=["wav"])
    if uploaded_file is not None:
        clear_outputs_folder(OUTPUTS_DIR)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            tmpf.write(uploaded_file.read())
            file_path = tmpf.name

        input_spec_path = os.path.join(OUTPUTS_DIR, "input_spec.png")
        input_features = get_audio_features(file_path, save_spectrogram_path=input_spec_path)
        st.markdown("**🎧 Nghe thử file truy vấn:**")
        st.audio(uploaded_file, format="audio/wav")
        st.image(input_spec_path, caption="Ảnh phổ truy vấn", use_container_width=True)

        st.markdown("**🔎 Đặc trưng truy vấn (gốc)**")
        feature_table = pd.DataFrame(
            [[input_features[k] for k in FEATURE_NAMES.keys()]],
            columns=[FEATURE_NAMES[k] for k in FEATURE_NAMES.keys()]
        ).T
        feature_table.columns = ["Giá trị"]
        st.dataframe(feature_table, use_container_width=True)

with right:
    if uploaded_file is not None:
        # Tải dữ liệu từ CSDL (sử dụng cache)
        db_df, means, stds = load_db_data()
        
        # Kết nối engine riêng cho việc ghi dữ liệu truy vấn (không cần cache)
        engine_write = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4")

        query_vec_raw = np.array([input_features[c] for c in feature_cols])
        query_vec_scaled = (query_vec_raw - means.values) / stds.values

        # Lưu đặc trưng truy vấn vào bảng features_queries
        query_data = {
            "filename": uploaded_file.name,
            "upload_time": datetime.now()
        }
        for i, col in enumerate(feature_cols):
            query_data[col] = query_vec_raw[i]
        for i, col in enumerate(scaled_cols):
            query_data[col] = query_vec_scaled[i]
        pd.DataFrame([query_data]).to_sql("features_queries", engine_write, if_exists="append", index=False)

        # So sánh cosine
        X_db_scaled = db_df[scaled_cols].values
        weights = np.array([0.3, 0.3, 0.2, 0.7, 0.9, 1.0, 1.0])  # theo thứ tự feature_cols
        sim_scores = [cosine_similarity_weighted(query_vec_scaled, x, weights) for x in X_db_scaled]

        db_df['cosine'] = sim_scores
        db_df_sorted = db_df.sort_values(by='cosine', ascending=False)
        top3 = db_df_sorted.head(3)

        st.markdown("### 🥇 **Top 3 file tương đồng nhất**")
        for _, row in top3.iterrows():
            col1, col2 = st.columns([1,2])
            with col1:
                st.markdown(f"**File:** `{row['filename']}`<br>**Nhạc cụ:** {row['label']}<br>**Độ tương đồng:** `{row['cosine']:.4f}`", unsafe_allow_html=True)
                file_in_db = os.path.join(DB_DIR, row["filename"])
                if os.path.exists(file_in_db):
                    st.audio(file_in_db, format="audio/wav")
                else:
                    st.warning("Không tìm thấy file gốc để phát thử.")
            with col2:
                spec_db = os.path.join(OUTPUTS_DIR, f"{row['filename']}_spec.png")
                if not os.path.isfile(spec_db):
                    try:
                        # Sử dụng hàm đã cache để trích xuất đặc trưng cho file từ DB
                        get_audio_features(file_in_db, save_spectrogram_path=spec_db)
                    except Exception as e: 
                        st.warning(f"Lỗi khi tạo ảnh phổ cho {row['filename']}: {e}")
                if os.path.exists(spec_db):
                    st.image(spec_db, caption="Ảnh phổ DB", use_container_width=True)

        try:
            os.remove(file_path)
            os.remove(input_spec_path)
        except:
            pass
