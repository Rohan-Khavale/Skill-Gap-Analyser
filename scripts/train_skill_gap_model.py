from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# App setup
# =====================
app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = Path(__file__).parent

# =====================
# Load ML artifacts
# =====================
VECTORIZER_PATH = BASE_DIR / "models/skill_vectorizer.joblib"
ROLE_MATRIX_PATH = BASE_DIR / "models/role_skill_matrix.joblib"
ROLES_PATH = BASE_DIR / "models/roles.joblib"

vectorizer = joblib.load(VECTORIZER_PATH)
role_matrix = joblib.load(ROLE_MATRIX_PATH)
roles = joblib.load(ROLES_PATH)

# =====================
# Helpers
# =====================
def extract_missing_skills(user_text, target_role, top_k=8):
    """
    ML-based skill gap detection using TF-IDF + cosine similarity.
    Returns ranked missing skills.
    """
    user_text = user_text.lower()
    target_role = target_role.lower()

    # Vectorize user input
    user_vec = vectorizer.transform([user_text])

    # Filter role rows
    role_indices = [i for i, r in enumerate(roles) if r == target_role]
    if not role_indices:
        return None, list(sorted(set(roles)))

    role_vecs = role_matrix[role_indices]

    # Find closest role profile
    sims = cosine_similarity(user_vec, role_vecs)
    best_idx = role_indices[int(np.argmax(sims))]

    # Extract important terms for that role
    feature_names = np.array(vectorizer.get_feature_names_out())
    role_weights = role_matrix[best_idx].toarray().flatten()

    # Top weighted skill terms
    top_terms = feature_names[role_weights.argsort()[::-1]][:top_k * 2]

    # Remove skills already mentioned by user
    missing = [
        term for term in top_terms
        if term not in user_text
        and len(term.split()) <= 3
    ]

    return missing[:top_k], None

# =====================
# Routes
# =====================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    user_text = f"""
    name: {data.get('name','')}
    status: {data.get('status','')}
    field: {data.get('field','')}
    skills: {data.get('skills','')}
    target role: {data.get('target','')}
    """

    target_role = data.get("target", "").lower()

    missing_skills, supported_roles = extract_missing_skills(
        user_text=user_text,
        target_role=target_role
    )

    if missing_skills is None:
        return jsonify({
            "error": "Target role not supported yet",
            "supported_roles": supported_roles
        })

    return jsonify({
        "target_role": target_role,
        "missing_skills": missing_skills,
        "status": "skill gap found" if missing_skills else "ready"
    })

@app.route("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)