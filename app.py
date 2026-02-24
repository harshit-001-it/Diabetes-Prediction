"""
src/app.py — Flask Web App for Diabetes Prediction
====================================================
Loads models/ artefacts and serves the web UI.
Templates and static assets live at the project root.
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

# ── Project-root-relative paths ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

# Load once at startup
model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json()
        features = [float(data[f]) for f in FEATURE_NAMES]
        arr      = np.array(features).reshape(1, -1)
        arr_sc   = scaler.transform(arr)
        pred     = int(model.predict(arr_sc)[0])

        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(arr_sc)[0][pred]) * 100
        elif hasattr(model, "decision_function"):
            raw  = model.decision_function(arr_sc)[0]
            prob = float(1 / (1 + np.exp(-raw))) * 100
        else:
            prob = 100.0 if pred == 1 else 0.0

        return jsonify({
            "prediction": "Diabetic" if pred == 1 else "Not Diabetic",
            "probability": round(prob, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model-info")
def model_info():
    results_path = os.path.join(MODELS_DIR, "model_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return jsonify(json.load(f))
    return jsonify([])


if __name__ == "__main__":
    app.run(debug=False, port=5000)
