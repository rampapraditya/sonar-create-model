import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
import numpy as np

# Load data yang sudah diproses
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
label_classes = np.load("label_encoder.npy", allow_pickle=True)  # Kelas label

# Ambil shape input
time_steps, n_mfcc = X_train.shape[1], X_train.shape[2]

# Membangun model LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, n_mfcc)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(len(label_classes), activation="softmax")  # Output sesuai jumlah kategori
])

# Kompilasi model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Melatih model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Simpan model
model.save("sonar_model.keras")

print("✅ Model berhasil dilatih dan disimpan sebagai sonar_model.keras!")
