import os
from datetime import datetime

from flask import (
    Flask, render_template, request, url_for, abort, send_from_directory, send_file
)
from werkzeug.utils import secure_filename

import firebase_admin
from firebase_admin import credentials, auth as fb_auth, messaging

from model import train_autoencoder, detect_anomalies

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont



app = Flask(__name__)

# ---------- Config ----------
UPLOAD_FOLDER = os.path.join("static", "uploads")
GRAPH_FOLDER = os.path.join("static", "graphs")
REPORT_FOLDER = os.path.join("static", "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


FIREBASE_WEB_CONFIG = {
    "apiKey": "AIzaSyBP2qiP9xOq2n-ueg48egQHK9qlteUMP5o",
    "authDomain": "anomaly-detection-in-cctv.firebaseapp.com",
    "projectId": "anomaly-detection-in-cctv",
    "storageBucket": "anomaly-detection-in-cctv.firebasestorage.app",
    "messagingSenderId": "1016994415144",
    "appId": "1:1016994415144:web:070ba602553c0c3c9f6b1d",
    "measurementId": "G-ZJXZ7PVV63",
}

SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise FileNotFoundError(
        f"Missing serviceAccountKey.json at: {SERVICE_ACCOUNT_PATH}. "
        f"Place it next to app.py (do NOT commit to GitHub)."
    )

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
    return send_from_directory(
        "static", "firebase-messaging-sw.js", mimetype="application/javascript"
    )


TOPIC = "alerts"


@app.post("/save-token")
def save_token():
    require_firebase_user()
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
    return render_template("register.html", firebase_config=FIREBASE_WEB_CONFIG)


@app.route("/")
def index():
    return render_template("index.html", firebase_config=FIREBASE_WEB_CONFIG)


@app.post("/upload")
def upload_video():
    decoded = require_firebase_user()
    uid = decoded.get("uid", "unknown")

    if "video" not in request.files:
        return "Error: No video file uploaded", 400

    file = request.files["video"]
    if file.filename == "":
        return "Error: No file selected", 400

    filename = secure_filename(file.filename)

    user_upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], uid)
    os.makedirs(user_upload_dir, exist_ok=True)

    saved_path = os.path.join(user_upload_dir, filename)
    file.save(saved_path)

    model = train_autoencoder(saved_path)

    base = os.path.splitext(filename)[0]
    user_graph_dir = os.path.join("graphs", uid)
    os.makedirs(os.path.join("static", user_graph_dir), exist_ok=True)

    graph_rel_path = os.path.join(user_graph_dir, f"{base}_anomaly_graph.png").replace("\\", "/")

    results = detect_anomalies(saved_path, model, graph_rel_path=graph_rel_path)
    if results is None:
        return "Error: Anomaly detection failed. Check your model.py function.", 500

    graph_path = results.get("graph_path", "")
    return render_template(
        "result.html",
        video_url=url_for("static", filename=f"uploads/{uid}/{filename}"),
        graph_url=url_for("static", filename=graph_path),
        timestamps=results.get("timestamps", []),
        threshold=results.get("threshold", 0.0),
        is_anomaly_detected=results.get("is_anomaly_detected", "No"),
        video_filename=filename,  # IMPORTANT for PDF payload
    )


@app.post("/generate-pdf")
def generate_pdf():
    decoded = require_firebase_user()
    uid = decoded.get("uid", "unknown")

    data = request.get_json() or {}
    filename = data.get("filename", "video.mp4")
    is_anomaly = data.get("is_anomaly_detected", "No")
    threshold = float(data.get("threshold", 0.0))
    timestamps = data.get("timestamps", [])
    graph_url = data.get("graph_url", "")

    base = os.path.splitext(filename)[0]
    pdf_filename = f"{base}_report.pdf"
    user_pdf_dir = os.path.join("static", "reports", uid)
    os.makedirs(user_pdf_dir, exist_ok=True)
    pdf_path = os.path.join(user_pdf_dir, pdf_filename)

    # --- Assets (put these files in static/assets/) ---
    top_logo_path = os.path.join("static", "assets", "logo.jpg")        # 2nd image
    cluster_logo_path = os.path.join("static", "assets", "Cluster_Hub.jpg")  # 3rd image

    # --- Styles (Times-Roman everywhere) ---
    base_styles = getSampleStyleSheet()
    style_h1 = ParagraphStyle("h1", parent=base_styles["Heading1"], fontName="Times-Roman", fontSize=20)
    style_h3 = ParagraphStyle("h3", parent=base_styles["Heading3"], fontName="Times-Roman", fontSize=13)
    style_n = ParagraphStyle("n", parent=base_styles["Normal"], fontName="Times-Roman", fontSize=11)

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 10)

        page_w, page_h = letter
        y_text = 0.85 * inch          # was ~0.55
        y_logo = y_text - 0.60 * inch

        canvas.drawCentredString(page_w / 2, y_text, "Powered by Cluster Hub")

        if os.path.exists(cluster_logo_path):
            logo_w = 0.60 * inch
            logo_h = 0.60 * inch
            canvas.drawImage(
                cluster_logo_path,
                (page_w - logo_w) / 2,
                y_text - 0.70 * inch,
                width=logo_w,
                height=logo_h,
                preserveAspectRatio=True,
                mask="auto",
            )

        canvas.restoreState()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        topMargin=0.7 * inch,
        bottomMargin=1.3 * inch,  # space for footer
    )

    elements = []

    # Top logo
    if os.path.exists(top_logo_path):
        elements.append(Image(top_logo_path, width=1 * inch, height=1 * inch))
        elements.append(Spacer(1, 0.15 * inch))

    # Title + metadata
    elements.append(Paragraph("Anomaly Detection Report", style_h1))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph(f"<b>File:</b> {filename}", style_n))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_n))
    elements.append(Paragraph(f"<b>Anomaly Detected:</b> {is_anomaly}", style_n))
    elements.append(Paragraph(f"<b>Threshold:</b> {threshold:.4f}", style_n))
    elements.append(Spacer(1, 0.2 * inch))

    # Table
    if timestamps:
        table_data = [["#", "Start (s)", "End (s)"]]
        for i, pair in enumerate(timestamps, 1):
            start = float(pair[0])
            end = float(pair[1])
            table_data.append([str(i), f"{start:.2f}", f"{end:.2f}"])

        table = Table(table_data)
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(Paragraph("<b>Detected Anomaly Intervals:</b>", style_h3))
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))
    else:
        elements.append(Paragraph("No anomalies detected.", style_n))
        elements.append(Spacer(1, 0.2 * inch))

    # Graph
    graph_filename = graph_url.split("/")[-1]
    graph_path = os.path.join("static", "graphs", uid, graph_filename)
    if graph_filename and os.path.exists(graph_path):
        elements.append(Paragraph("<b>Anomaly Graph:</b>", style_h3))
        elements.append(Image(graph_path, width=6 * inch, height=3.2 * inch))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)

    return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)


if __name__ == "__main__":
    app.run(debug=True)
