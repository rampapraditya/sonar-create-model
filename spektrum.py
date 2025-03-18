import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Baca file WAV
file_path = "dataset/tangker/tangker1.wav"  # Ganti dengan path file WAV kamu
sample_rate, audio_data = wav.read(file_path)

# Jika stereo, ubah ke mono dengan rata-rata dua kanal
if len(audio_data.shape) > 1:
    audio_data = np.mean(audio_data, axis=1)

# Normalisasi data PCM (16-bit ke float)
audio_data = audio_data / np.max(np.abs(audio_data))

# Hitung FFT
n = len(audio_data)
frequencies = np.fft.rfftfreq(n, d=1/sample_rate)  # Sumbu frekuensi
fft_magnitudes = np.abs(np.fft.rfft(audio_data))  # Besar FFT

# Plot Spektrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, fft_magnitudes, color="blue")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude")
plt.title("Spektrum Suara")
plt.grid()
plt.show()
