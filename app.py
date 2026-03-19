import os
import traceback
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from utils.feature_extractor import extract_features

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "advanced_model_50k.pkl")

# Load model at startup
model = None
try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model at {MODEL_PATH}: {e}")
    model = None


def interpret_prediction(pred):
    """
    Model training:
        0 -> Phishing
        1 -> Legitimate
    """
    if isinstance(pred, (list, tuple, pd.Series)):
        pred = pred[0]

    try:
        pred = int(pred)
        if pred == 0:
            return "Phishing"
        elif pred == 1:
            return "Legitimate"
        else:
            return "Unknown"
    except Exception:
        return "Unknown"


@app.route("/")
def index():
    return render_template("index.html", site_name="PhishHunt")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Invalid request — JSON with 'url' required."}), 400

    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Empty URL provided."}), 400

    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url

    try:
        feats = extract_features(url)
        X = pd.DataFrame([feats])

        pred_raw = model.predict(X)
        label = interpret_prediction(pred_raw)

        # Confidence score
        confidence = "N/A"
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = f"{max(proba) * 100:.2f}%"
            elif hasattr(model, "decision_function"):
                import math
                score = float(model.decision_function(X)[0])
                prob_like = 1 / (1 + math.exp(-score))
                confidence = f"{prob_like * 100:.2f}%"
        except Exception:
            pass

        return jsonify({"result": label, "confidence": confidence}), 200

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Prediction error: {e}\n{tb}")
        return jsonify({"error": "Failed to generate prediction.", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)