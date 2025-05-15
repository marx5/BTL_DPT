import streamlit as st
import os
import pandas as pd
import tempfile
from audio_features import extract_audio_features
import joblib
import numpy as np
import glob

OUTPUTS_DIR = "outputs"
DB_CSV = "audio_features_db.csv"
SCALER_PATH = "scaler_save.pkl"
DB_DIR = "audio_db"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Hàm xoá toàn bộ file trong outputs/
def clear_outputs_folder(outputs_dir):
    for file in glob.glob(os.path.join(outputs_dir, "*")):
        try:
            os.remove(file)
        except Exception as e:
            pass  # hoặc in(f"Lỗi xoá {file}: {e}")

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    num = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    return num / denom if denom != 0 else 0

FEATURE_NAMES = {
    "energy": "Năng lượng",
    "zero_crossing_rate": "Tốc độ đổi dấu",
    "percent_silence": "Khoảng lặng (%)",
    "bandwidth": "Băng thông (Hz)",
    "spectral_centroid": "Trung tâm phổ (Hz)",
    "harmonicity": "Độ điều hòa (dB)",
    "pitch": "Độ cao (Hz)"
}

st.set_page_config(page_title="Truy vấn nhạc cụ bộ hơi", layout="wide")
st.title("🎷 Hệ thống truy vấn & nhận dạng nhạc cụ bộ hơi")

with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
    st.markdown(
        """
        1. Tải file âm thanh truy vấn (.wav) phía trái.
        2. Hệ thống trích xuất đặc trưng, hiển thị bảng so sánh với cơ sở dữ liệu.
        3. Nghe thử, xem ảnh phổ và tra cứu top 3 file giống nhất.
        4. Độ tương đồng tính bằng Cosine trên đặc trưng đã chuẩn hóa.
        """
    )

# LAYOUT: 2 cột
left, right = st.columns([1,2])

with left:
    uploaded_file = st.file_uploader("🎼 Chọn file .wav truy vấn", type=["wav"])
    if uploaded_file is not None:
        # Xoá toàn bộ file trong outputs/ mỗi lần nhận dạng mới
        clear_outputs_folder(OUTPUTS_DIR)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
            tmpf.write(uploaded_file.read())
            file_path = tmpf.name

        input_spec_path = os.path.join(OUTPUTS_DIR, "input_spec.png")
        input_features = extract_audio_features(file_path, save_spectrogram_path=input_spec_path)
        st.markdown("**🎧 Nghe thử file truy vấn:**")
        st.audio(uploaded_file, format="audio/wav")
        st.image(input_spec_path, caption="Ảnh phổ truy vấn", use_container_width=True)
        # Hiển thị bảng đặc trưng truy vấn
        st.markdown("**🔎 Đặc trưng truy vấn**")
        feature_table = pd.DataFrame(
            [[input_features[k] for k in FEATURE_NAMES.keys()]],
            columns=[FEATURE_NAMES[k] for k in FEATURE_NAMES.keys()]
        ).T
        feature_table.columns = ["Giá trị"]
        st.dataframe(feature_table, use_container_width=True)

with right:
    if uploaded_file is not None:
        if not os.path.isfile(DB_CSV) or not os.path.isfile(SCALER_PATH):
            st.error("Chưa có file audio_features_db.csv hoặc scaler_save.pkl! Hãy chạy script trích xuất đặc trưng trước.")
        else:
            df = pd.read_csv(DB_CSV)
            scaler = joblib.load(SCALER_PATH)
            feature_cols = list(FEATURE_NAMES.keys())
            X_db = df[feature_cols].values
            query_vec_df = pd.DataFrame([[input_features[c] for c in feature_cols]], columns=feature_cols)
            query_vec_scaled = scaler.transform(query_vec_df)[0]
            sim_scores = []
            for i, row in df.iterrows():
                db_vec = X_db[i]
                sim = cosine_similarity(query_vec_scaled, db_vec)
                sim_scores.append({
                    "filename": row["filename"],
                    "label": row.get("label", ""),
                    "cosine": sim,
                    **{k: row[k] for k in feature_cols}
                })
            results_df = pd.DataFrame(sim_scores)
            results_df = results_df.sort_values(by="cosine", ascending=False)
            top3 = results_df.head(3)

            # st.markdown("### 📊 **So sánh toàn bộ cơ sở dữ liệu** (bấm vào bảng để xem chi tiết)")
            # show_cols = ["filename", "label", "cosine"] + list(FEATURE_NAMES.keys())
            # results_show = results_df[show_cols]
            # results_show = results_show.rename(columns={
            #     "filename": "File",
            #     "label": "Nhạc cụ",
            #     "cosine": "Độ tương đồng (Cosine)",
            #     **FEATURE_NAMES
            # })
            # st.dataframe(results_show.style.highlight_max(subset=["Độ tương đồng (Cosine)"], color="lightgreen"), height=340)

            st.markdown("### 🥇 **Top 3 file tương đồng nhất**")
            for idx, row in top3.iterrows():
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
                            extract_audio_features(file_in_db, save_spectrogram_path=spec_db)
                        except: pass
                    if os.path.exists(spec_db):
                        st.image(spec_db, caption="Ảnh phổ DB", use_container_width=True)

        # Xoá file tạm
        try:
            os.remove(file_path)
            os.remove(input_spec_path)
        except:
            pass
