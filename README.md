# HỆ THỐNG TRÍCH XUẤT, LƯU TRỮ, TRUY VẤN & ĐÁNH GIÁ ĐẶC TRƯNG ÂM THANH NHẠC CỤ BỘ HƠI

## 1. Mục đích

* Trích xuất đặc trưng âm thanh chuẩn lý thuyết cho file nhạc cụ bộ hơi (sáo, clarinet, saxophone, trumpet...).
* Lưu trữ đặc trưng đã **chuẩn hóa** vào file CSV phục vụ tìm kiếm, nhận dạng, phân tích hoặc huấn luyện.
* Truy vấn qua **giao diện web**: upload file, xem đặc trưng, nghe thử, so sánh, hiển thị file tương tự nhất (độ tương đồng bằng Cosine trên đặc trưng đã chuẩn hóa).
* Đánh giá độ chính xác hệ thống với thư mục test độc lập.

---

## 2. Cấu trúc thư mục

```
multimedia_audio_search/
│
├── audio_features.py               # Module trích xuất đặc trưng
├── extract_features_to_csv.py      # Script tạo file đặc trưng CSV, scaler
├── app.py                          # Giao diện web truy vấn, so sánh
├── evaluate_accuracy.py            # Script đánh giá độ chính xác với bộ test (tuỳ chọn)
├── requirements.txt                # Danh sách thư viện cần cài
├── README.md                       # Hướng dẫn chi tiết (file này)
│
├── audio_db/                       # Thư mục chứa file .wav nhạc cụ (train/CSLD)
│     └── flute_1.wav, clarinet_1.wav, ...
├── audio_test/                     # Thư mục file test đã gắn nhãn (đánh giá hệ thống)
│     └── flute_test01.wav, ...
├── outputs/                        # Chứa ảnh phổ spectrogram (auto dọn dẹp mỗi lượt truy vấn)
├── audio_features_db.csv           # File đặc trưng sinh ra sau khi trích xuất
├── scaler_save.pkl                 # File scaler (chuẩn hóa) sinh ra khi huấn luyện
└── venv/                           # Môi trường ảo Python (tạo khi setup)
```

---

## 3. Hướng dẫn cài đặt môi trường (venv nội bộ)

**Bước 1:** Mở terminal/cmd, chuyển tới thư mục dự án
**Bước 2:** Tạo môi trường ảo (venv)

```bash
python -m venv venv
```

**Bước 3:** Kích hoạt môi trường ảo

* **Windows:**

  ```
  venv/Scripts/activate
  ```
* **MacOS/Linux:**

  ```
  source venv/bin/activate
  ```

**Bước 4:** Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

---

## 4. Hướng dẫn sử dụng hệ thống

### 4.1. Chuẩn bị dữ liệu

* Đặt file `.wav` của từng nhạc cụ vào thư mục `audio_db/` (huấn luyện/cơ sở dữ liệu).
* Đặt file `.wav` để kiểm thử vào `audio_test/`. **Tên file nên chứa nhãn**, ví dụ: `flute_001.wav`, `trumpet_test01.wav`.

### 4.2. Trích xuất đặc trưng, chuẩn hóa và sinh file CSV

```bash
python extract_features_to_csv.py
```

* File đặc trưng **chuẩn hóa**: `audio_features_db.csv`
* File scaler (dùng cho truy vấn): `scaler_save.pkl`

### 4.3. Khởi động giao diện truy vấn

```bash
streamlit run app.py
```

* Giao diện web sẽ mở ra tại [http://localhost:8501](http://localhost:8501).
* **Tính năng:**

  * Upload file `.wav` truy vấn (hệ thống sẽ dọn sạch ảnh phổ cũ trong `outputs/`).
  * Xem đặc trưng, nghe thử, so sánh với tất cả file trong cơ sở dữ liệu.
  * Bảng so sánh chi tiết, top 3 file tương đồng nhất, xem ảnh phổ, nghe lại từng file.

### 4.4. Đánh giá hệ thống với bộ kiểm thử

* Đặt file `.wav` test vào `audio_test/` (đã gắn nhãn đúng).
* Chạy script đánh giá:

```bash
python evaluate_accuracy.py
```

* Kết quả:

  * Hiện số nhận dạng đúng/sai và độ chính xác top-1.
  * Ghi file `test_accuracy_results.csv` (xem chi tiết từng file, cosine, nhãn thực/dự đoán).

---

## 5. Các đặc trưng âm thanh sử dụng

* **Năng lượng trung bình (energy):** Độ lớn tổng thể tín hiệu.
* **Tốc độ đổi dấu (zero\_crossing\_rate):** Đặc trưng tần số trung bình.
* **Phần trăm khoảng lặng (percent\_silence):** Tỷ lệ thời gian âm thanh nhỏ/nghỉ.
* **Băng thông (bandwidth):** Dải tần số hoạt động của âm thanh.
* **Trung tâm phổ (spectral\_centroid):** Độ sáng của âm thanh.
* **Độ điều hòa âm (harmonicity):** Đánh giá độ “sạch” âm thanh, ít nhiễu.
* **Độ cao thấp (pitch):** Tần số cơ bản.
* **Ảnh phổ (spectrogram):** Hình ảnh trực quan phổ tần số theo thời gian (chỉ để quan sát).

---

## 6. Yêu cầu thư viện (requirements.txt)

```
librosa
numpy
matplotlib
pandas
streamlit
scikit-learn
joblib
```

---
