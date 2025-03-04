import os

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path dataset
DATASET_PATH = "dataset/"
CATEGORIES = ["cargo", "nelayan", "tangker"]
SAMPLE_RATE = 16000  # Standarisasi ke 16kHz
N_MFCC = 20  # Jumlah koefisien MFCC

def pad_sequences_alternative(sequences, maxlen, padding='post'):
    padded = np.zeros((len(sequences), maxlen, sequences[0].shape[1]), dtype=np.float32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        if padding == 'post':
            padded[i, :length, :] = seq[:length]
        else:  # padding = 'pre'
            padded[i, -length:, :] = seq[:length]
    return padded

def extract_features():
    X, y, lengths = [], [], []

    for category in CATEGORIES:
        folder_path = os.path.join(DATASET_PATH, category)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Load file audio
            y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Ekstrak fitur MFCC
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
            mfcc = mfcc.T  # Transpos agar shape (time_steps, n_mfcc)

            # Simpan panjang untuk analisis
            lengths.append(mfcc.shape[0])

            X.append(mfcc)
            y.append(category)

    # Tentukan max_length berdasarkan distribusi data
    max_length = int(np.percentile(lengths, 95))  # 95% dari data masuk

    # Normalisasi panjang dengan padding atau cropping
    X = pad_sequences_alternative(X, max_length)

    # Encode label menjadi angka
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Bagi data menjadi training & testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simpan hasil preprocessing
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("label_encoder.npy", label_encoder.classes_)

    print(f"✅ Preprocessing selesai! Data disimpan dengan shape {X_train.shape}.")
    print(f"🔍 Panjang MFCC rata-rata: {np.mean(lengths)}, Maksimal: {max_length}")


# Jalankan preprocessing
extract_features()
