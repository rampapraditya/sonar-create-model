import librosa
import numpy as np
import tensorflow as tf
from preprocess import pad_sequences_alternative, SAMPLE_RATE, N_MFCC

# Load model
model = tf.keras.models.load_model("sonar_model.keras")

# Load label encoder
label_classes = np.load("label_encoder.npy", allow_pickle=True)

def predict_audio(file_path):
    """
    Memprediksi jenis kapal berdasarkan suara sonar.
    """
    y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T  # Transpos agar shape (time_steps, n_mfcc)

    # Load max_length dari hasil preprocessing
    X_train = np.load("X_train.npy")
    max_length = X_train.shape[1]

    # Padding agar panjangnya sama dengan data training
    mfcc_padded = pad_sequences_alternative([mfcc], maxlen=max_length)

    # Prediksi
    prediction = model.predict(mfcc_padded)
    predicted_label = np.argmax(prediction)

    return label_classes[predicted_label]

# Contoh penggunaan
file_path = "dataset/cargo/cargo6.wav"  # Ubah sesuai file yang ingin diuji
hasil = predict_audio(file_path)
print(f"Hasil prediksi: {hasil}")
