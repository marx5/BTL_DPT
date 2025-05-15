import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_audio_features(file_path, frame_length=2048, hop_length=512,
                          silence_threshold=0.02, silence_min_duration=0.2,
                          save_spectrogram_path=None):
    """
    Trích xuất các đặc trưng âm thanh đúng tài liệu thầy Hóa.
    """
    y, sr = librosa.load(file_path, sr=None)
    N = len(y)

    # 1. Năng lượng trung bình
    energy = np.mean(y ** 2)

    # 2. Tốc độ đổi dấu tín hiệu (Zero Crossing Rate)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length))

    # 3. Phần trăm khoảng lặng
    abs_y = np.abs(y)
    silence_mask = abs_y < silence_threshold
    num_silence_samples = int(silence_min_duration * sr)
    silent_segments = []
    current_start = None
    for i, val in enumerate(silence_mask):
        if val and current_start is None:
            current_start = i
        elif not val and current_start is not None:
            if i - current_start >= num_silence_samples:
                silent_segments.append((current_start, i))
            current_start = None
    if current_start is not None and N - current_start >= num_silence_samples:
        silent_segments.append((current_start, N))
    total_silence_samples = sum(seg[1] - seg[0] for seg in silent_segments)
    percent_silence = total_silence_samples / N

    # 4. Băng thông (Bandwidth)
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    mean_spectrum = np.mean(S, axis=1)
    db_threshold = np.max(mean_spectrum) / (10 ** (3 / 20))
    nonzero_freqs = freqs[mean_spectrum >= db_threshold]
    bandwidth = nonzero_freqs[-1] - nonzero_freqs[0] if len(nonzero_freqs) > 0 else 0

    # 5. Phân bố năng lượng âm thanh (Spectral Centroid/Brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length))

    # 6. Độ điều hòa âm (Harmonicity) — dùng Harmonic-to-Noise Ratio (HNR)
    harmonic_y = librosa.effects.harmonic(y)
    noise_y = y - harmonic_y
    power_harm = np.sum(harmonic_y ** 2)
    power_noise = np.sum(noise_y ** 2)
    harmonicity = 10 * np.log10((power_harm + 1e-9) / (power_noise + 1e-9))  # dB

    # 7. Độ cao thấp của âm thanh (Pitch)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch = np.median(pitch_values) if len(pitch_values) > 0 else 0

    # 8. Ảnh phổ (Spectrogram)
    if save_spectrogram_path:
        plt.figure(figsize=(8, 3))
        D = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(save_spectrogram_path, bbox_inches='tight')
        plt.close()

    features = {
        "energy": energy,
        "zero_crossing_rate": zcr,
        "percent_silence": percent_silence,
        "bandwidth": bandwidth,
        "spectral_centroid": spectral_centroid,
        "harmonicity": harmonicity,
        "pitch": pitch,
    }
    return features
