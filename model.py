import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def build_autoencoder():
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 1)))
    model.add(MaxPooling2D((2, 2), padding="same"))
    model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), padding="same"))

    # Decoder
    model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))

    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(video_path, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        frames.append(gray)

        if len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        return build_autoencoder()

    frames = np.array(frames, dtype="float32") / 255.0
    frames = np.expand_dims(frames, axis=-1)

    autoencoder = build_autoencoder()
    autoencoder.fit(frames, frames, epochs=3, batch_size=8, verbose=1)
    return autoencoder


def detect_anomalies(video_path, autoencoder, threshold=0.01, graph_rel_path="anomaly_graph.png"):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    # analyze 1 frame every 2 seconds
    frame_interval = max(int(fps * 2), 1)

    frame_count = 0
    anomaly_frames = []
    errors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))

            gray_norm = gray.astype("float32") / 255.0
            gray_norm = np.expand_dims(gray_norm, axis=(0, -1))

            reconstructed = autoencoder.predict(gray_norm, verbose=0)
            loss = float(np.mean((gray_norm - reconstructed) ** 2))
            errors.append(loss)

            if loss > threshold:
                anomaly_frames.append(frame_count)

        frame_count += 1

    cap.release()

    # Convert frame indices to (start, end) seconds
    timestamps = []
    if anomaly_frames:
        start = anomaly_frames[0]
        for i in range(1, len(anomaly_frames)):
            if anomaly_frames[i] - anomaly_frames[i - 1] > frame_interval:
                end = anomaly_frames[i - 1]
                timestamps.append((round(start / fps, 2), round(end / fps, 2)))
                start = anomaly_frames[i]
        timestamps.append((round(start / fps, 2), round(anomaly_frames[-1] / fps, 2)))

    # Save error graph
    os.makedirs("static", exist_ok=True)
    full_graph_path = os.path.join("static", graph_rel_path).replace("\\", "/")
    os.makedirs(os.path.dirname(full_graph_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    x_values = np.arange(len(errors)) * 2  # seconds
    plt.plot(x_values, errors, color="orange", label="Reconstruction Error")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.title("Anomaly Detection Graph (1 frame per 2 seconds)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.savefig(full_graph_path)
    plt.close()

    return {
        "timestamps": timestamps,
        "graph_path": graph_rel_path.replace("\\", "/"),
        "threshold": threshold,
        "is_anomaly_detected": "Yes" if len(timestamps) > 0 else "No",
    }
