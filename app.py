import os
from flask import Flask, render_template, request, url_for, abort, redirect, send_from_directory
from werkzeug.utils import secure_filename

import firebase_admin
from firebase_admin import credentials, auth as fb_auth, messaging

from model import train_autoencoder, detect_anomalies

app = Flask(__name__)

# ---------- Config ----------
UPLOAD_FOLDER = os.path.join("static", "uploads")
GRAPH_FOLDER = os.path.join("static", "graphs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Put your Firebase Web config here ONCE (shared by login + index templates)
FIREBASE_WEB_CONFIG = {
  "apiKey": "AIzaSyBP2qiP9xOq2n-ueg48egQHK9qlteUMP5o",
  "authDomain": "anomaly-detection-in-cctv.firebaseapp.com",
  "projectId": "anomaly-detection-in-cctv",
  "storageBucket": "anomaly-detection-in-cctv.firebasestorage.app",
  "messagingSenderId": "1016994415144",
  "appId": "1:1016994415144:web:070ba602553c0c3c9f6b1d",
  "measurementId": "G-ZJXZ7PVV63",  # optional
}


# ---------- Firebase Admin init (Option B: path relative to this file) ----------
SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise FileNotFoundError(
        f"Missing serviceAccountKey.json at: {SERVICE_ACCOUNT_PATH}. "
        f"Place it next to app.py (do NOT commit to GitHub)."
    )

# Avoid double-init when Flask debug reloads
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)


def require_firebase_user():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        abort(401, description="Missing Bearer token")

    id_token = auth_header.split("Bearer ", 1)[1].strip()
    if not id_token:
        abort(401, description="Empty Bearer token")

    try:
        return fb_auth.verify_id_token(id_token)
    except Exception:
        abort(401, description="Invalid/expired token")

@app.route("/firebase-messaging-sw.js")
def fcm_sw():
    return send_from_directory("static", "firebase-messaging-sw.js", mimetype="application/javascript")

TOPIC = "alerts"

@app.post("/save-token")
def save_token():
    require_firebase_user()  # verify user logged in
    token = (request.get_json() or {}).get("token")
    if not token:
        return {"error": "Missing token"}, 400

    resp = messaging.subscribe_to_topic([token], TOPIC)
    return {"ok": True, "success": resp.success_count}


@app.route("/login")
def login_page():
    return render_template("login.html", firebase_config=FIREBASE_WEB_CONFIG)

@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/")
def index():
    # upload page (client JS redirects to /login if not logged in)
    return render_template("index.html", firebase_config=FIREBASE_WEB_CONFIG)


@app.route("/upload", methods=["POST"])
def upload_video():
    decoded = require_firebase_user()
    uid = decoded.get("uid", "unknown")

    if "video" not in request.files:
        return "Error: No video file uploaded", 400

    file = request.files["video"]
    if file.filename == "":
        return "Error: No file selected", 400

    filename = secure_filename(file.filename)

    # Store uploads per user
    user_upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], uid)
    os.makedirs(user_upload_dir, exist_ok=True)
    saved_path = os.path.join(user_upload_dir, filename)
    file.save(saved_path)

    # Train + detect
    model = train_autoencoder(saved_path)

    # Save graph per user+file (avoids overwrite)
    base = os.path.splitext(filename)[0]
    user_graph_dir = os.path.join("graphs", uid)
    os.makedirs(os.path.join("static", user_graph_dir), exist_ok=True)
    graph_rel_path = os.path.join(user_graph_dir, f"{base}_anomaly_graph.png").replace("\\", "/")

    results = detect_anomalies(saved_path, model, graph_rel_path=graph_rel_path)
    if results is None:
        return "Error: Anomaly detection failed. Check your model.py function.", 500

    return render_template(
        "result.html",
        video_url=url_for("static", filename=f"uploads/{uid}/{filename}"),
        graph_url=url_for("static", filename=results.get("graph_path", "")),
        timestamps=results.get("timestamps", []),
        threshold=results.get("threshold", 0.0),
        is_anomaly_detected=results.get("is_anomaly_detected", "No"),
    )

if __name__ == "__main__":
    app.run(debug=True)
