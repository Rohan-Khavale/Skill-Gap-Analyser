# app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from pathlib import Path
from flask_cors import CORS

MODEL_PATH = Path("models/demo_pipeline.joblib")
if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run scripts/train_and_save_demo.py first.")

pipeline = joblib.load(MODEL_PATH)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
from flask import send_from_directory

# helper to compute top tokens for prediction (linear model)
def explain_prediction(text, top_k=6):
    # vectorizer and classifier from pipeline
    vect = pipeline.named_steps[next(k for k in pipeline.named_steps if 'tfidf' in k.lower() or 'vector' in k.lower())]
    clf = pipeline.named_steps[next(k for k in pipeline.named_steps if 'logistic' in k.lower() or 'clf' in k.lower())]

    X = vect.transform([text])
    # coef shape: (n_classes, n_features) for multiclass; for binary sklearn returns (1, n_features)
    coefs = clf.coef_
    if coefs.shape[0] == 1:
        coefs = coefs[0]
    else:
        # pick class 1 coefs if multi
        coefs = coefs[1]

    # get tokens present in the input
    indices = X.nonzero()[1]
    token_names = np.array(vect.get_feature_names_out())
    token_scores = coefs[indices]
    # sort by absolute contribution
    order = np.argsort(-np.abs(token_scores))[:top_k]
    toks = token_names[indices][order].tolist()
    scores = token_scores[order].tolist()
    return list(zip(toks, [float(s) for s in scores]))

@app.route("/predict", methods=["POST"])
def predict():
    js = request.json or {}
    text = js.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "empty input"}), 400

    pred = pipeline.predict([text])[0]
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0].tolist()

    explanation = explain_prediction(text, top_k=6)

    return jsonify({
        "text": text,
        "prediction": int(pred),
        "probabilities": probs,
        "explanation": explanation
    })

# serve frontend at root
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    # dev server on port 8000
    app.run(host="0.0.0.0", port=8000, debug=False)

@app.route("/healthz")
def healthz():
    return "ok", 200