from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import uuid
import sqlite3
import subprocess
import sys
from PIL import Image

from model import CancerCNN
from utils import preprocess
from database import init_db

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)  # allow React (localhost:3000)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL PATHS ----------------
MODEL_PATHS = {
    "oral": "models/oral_model.pth",
    "eye": "models/eye_model.pth",
    "skin": "models/skin_model.pth"
}

loaded_models = {}

def load_model(model_type):
    if model_type not in loaded_models:
        model = CancerCNN()
        model.load_state_dict(
            torch.load(MODEL_PATHS[model_type], map_location="cpu")
        )
        model.eval()
        loaded_models[model_type] = model
    return loaded_models[model_type]

# ---------------- DATABASE INIT ----------------
init_db()

# ---------------- HEALTH CHECK ----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Backend running"})

# ---------------- IMAGE PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        cancer_type = request.form.get("type")
        if cancer_type not in MODEL_PATHS:
            return jsonify({"error": "Invalid cancer type"}), 400

        image_file = request.files["image"]

        # Save uploaded image
        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        img = preprocess(image)

        # Load model and predict (probability is for CANCER class)
        model = load_model(cancer_type)
        with torch.no_grad():
            logits = model(img)
            prob = torch.sigmoid(logits).item()  # 0–1

        # Use a slightly higher threshold to reduce false positives
        prediction = "Cancer Detected" if prob > 0.7 else "No Cancer Detected"

        # Save to database
        conn = sqlite3.connect("cancer_data.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO records (cancer_type, image_path, prediction, confidence)
            VALUES (?, ?, ?, ?)
        """, (cancer_type, image_path, prediction, prob))
        conn.commit()
        conn.close()

        return jsonify({
            "prediction": prediction,
            "confidence": round(prob, 4)  # frontend converts to %
        })

    except Exception as e:
        print("❌ Predict error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- WEBCAM TRIGGER ----------------
@app.route("/webcam", methods=["GET"])
def webcam():
    try:
        python_exe = sys.executable          # venv python
        script_path = os.path.abspath("webcam_oral.py")

        print("🚀 Launching webcam:")
        print("Python:", python_exe)
        print("Script:", script_path)

        subprocess.Popen(
            [python_exe, script_path],
            shell=False
        )

        return jsonify({"status": "Webcam started"}), 200

    except Exception as e:
        print("❌ Webcam launch failed:", e)
        return jsonify({"error": str(e)}), 500

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)