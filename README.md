# HỆ THỐNG TRÍCH XUẤT, LƯU TRỮ, TRUY VẤN & ĐÁNH GIÁ ĐẶC TRƯNG ÂM THANH NHẠC CỤ BỘ HƠI

---

## 1. Mục đích

* Trích xuất đặc trưng âm thanh cho file nhạc cụ bộ hơi.
* Lưu trữ đặc trưng (cả **gốc** và **chuẩn hóa**) vào MySQL.
* Truy vấn, nhận dạng âm thanh, so sánh cosine trực tiếp bằng dữ liệu MySQL.
* Đánh giá độ chính xác nhận dạng.

---

## 2. Cấu trúc thư mục

```
multimedia_audio_search/
│
├── audio_features.py                   # Module trích xuất đặc trưng
├── extract_features.py                 # Script tạo đặc trưng, chuẩn hóa, lưu vào MySQL (tự động tạo db/bảng, xóa dữ liệu cũ)
├── app.py                              # Giao diện web truy vấn, nhận dạng (dùng MySQL)
├── test_accuracy.py                    # Đánh giá độ chính xác với bộ test (dùng MySQL)
├── requirements.txt                    # Danh sách thư viện cần cài
├── README.md                           # Hướng dẫn chi tiết (file này)
│
├── audio_db/                           # Thư mục chứa file .wav nhạc cụ (train/CSLD)
│     └── flute_1.wav, clarinet_1.wav, ...
├── audio_test/                         # Thư mục file test đã gắn nhãn (đánh giá hệ thống)
│     └── flute_test01.wav, ...
├── outputs/                            # Chứa ảnh phổ spectrogram (tự động dọn dẹp mỗi truy vấn)
└── venv/                               # Môi trường ảo Python (tạo khi setup)
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
  * Command Prompt (cmd):
    ```
    venv\Scripts\activate.bat
    ```
  * PowerShell:
    ```
    .\venv\Scripts\Activate.ps1
    ```
  * Git Bash:
    ```
    source venv/Scripts/activate
    ```

* **MacOS/Linux:**
  * Bash/Zsh:
    ```
    source venv/bin/activate
    ```
  * Fish:
    ```
    source venv/bin/activate.fish
    ```
  * Csh:
    ```
    source venv/bin/activate.csh
    ```

**Bước 4:** Cài đặt thư viện cần thiết

```bash
pip install -r requirements.txt
```

---

## 4. Hướng dẫn sử dụng hệ thống

### 4.1. Chuẩn bị dữ liệu

* Đặt file `.wav` từng nhạc cụ vào `audio_db/` (huấn luyện/cơ sở dữ liệu).
* Đặt file `.wav` kiểm thử vào `audio_test/`. **Tên file nên chứa nhãn**, ví dụ: `flute_001.wav`, `trumpet_test01.wav`.

### 4.2. Tạo database, bảng, và sinh đặc trưng, lưu vào MySQL (tự động)

```bash
python extract_features_to_mysql.py
```

* Tự động tạo database, bảng, **xóa toàn bộ dữ liệu cũ** và **thêm dữ liệu mới** vào bảng mỗi khi chạy.

### 4.3. Khởi động giao diện truy vấn (web)

```bash
streamlit run app.py
```

* Upload file `.wav` truy vấn. Hệ thống tự động chuẩn hóa đặc trưng truy vấn theo dữ liệu trong MySQL, so sánh cosine trực tiếp với DB, hiển thị kết quả tốt nhất, nghe thử, xem ảnh phổ.

### 4.4. Đánh giá độ chính xác trên bộ kiểm thử

```bash
python evaluate_accuracy_from_mysql.py
```

* Kết quả:

  * In số lần nhận dạng đúng/sai, tính độ chính xác top-1.
  * Ghi chi tiết từng file test vào `test_accuracy_results.csv`.

---

## 5. Các đặc trưng âm thanh sử dụng

* **Năng lượng trung bình (energy)**
* **Tốc độ đổi dấu (zero\_crossing\_rate)**
* **Phần trăm khoảng lặng (percent\_silence)**
* **Băng thông (bandwidth)**
* **Trung tâm phổ (spectral\_centroid)**
* **Độ điều hòa âm (harmonicity)**
* **Độ cao thấp (pitch)**
* **Ảnh phổ (spectrogram)**: Hình ảnh trực quan, để quan sát.

> **Tất cả đặc trưng đều được lưu song song bản gốc và bản đã chuẩn hóa (\_scaled) vào bảng MySQL.**

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
pymysql
sqlalchemy
```

---



## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request.

## License

MIT License