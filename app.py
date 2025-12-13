
from flask import Flask, request, jsonify, send_from_directory
import joblib
from pathlib import Path

app = Flask(__name__, static_folder="static")

MODEL_PATH = Path("models/demo_pipeline.joblib")
pipeline = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    text = f"""
    Name: {data.get('name')}
    Status: {data.get('status')}
    Field: {data.get('field')}
    Skills: {data.get('skills')}
    Target: {data.get('target')}
    """

    result = pipeline.predict([text])[0]

    return jsonify({
        "result": result
    })

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
