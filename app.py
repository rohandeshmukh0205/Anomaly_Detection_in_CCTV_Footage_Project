from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os

from model import train_autoencoder, detect_anomalies

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "Error: No video file uploaded", 400

    file = request.files["video"]
    if file.filename == "":
        return "Error: No file selected", 400

    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(saved_path)

    model = train_autoencoder(saved_path)
    results = detect_anomalies(saved_path, model)

    if results is None:
        return "Error: Anomaly detection failed. Check your model.py function.", 500

    return render_template(
        "result.html",
        video_url=url_for("static", filename=f"uploads/{filename}"),
        timestamps=results.get("timestamps", []),
        graph_url=url_for("static", filename=results.get("graph_path", "")),
        threshold=results.get("threshold", 0.0),
        is_anomaly_detected=results.get("is_anomaly_detected", "No"),
    )


if __name__ == "__main__":
    app.run(debug=True)
