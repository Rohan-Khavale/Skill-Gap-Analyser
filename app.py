from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

ROLE_SKILLS = {
    "software engineer": [
        "data structures", "algorithms", "python", "git", "system design"
    ],
    "backend developer": [
        "python", "databases", "apis", "sql", "system design"
    ],
    "frontend developer": [
        "html", "css", "javascript", "react", "ui/ux"
    ],
    "data scientist": [
        "python", "statistics", "machine learning", "pandas", "sql"
    ],
    "devops": [
        "linux", "docker", "kubernetes", "ci/cd", "cloud"
    ],
    "investment banker": [
        "financial modeling", "valuation", "excel", "accounting", "powerpoint"
    ]
}

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/healthz")
def healthz():
    return "OK", 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    user_skills = data.get("skills", "").lower()
    target_role = data.get("target", "").lower()

    if target_role not in ROLE_SKILLS:
        return jsonify({
            "error": "Target role not supported",
            "supported_roles": list(ROLE_SKILLS.keys())
        })

    user_skill_list = [s.strip() for s in user_skills.split(",") if s.strip()]
    required_skills = ROLE_SKILLS[target_role]

    missing_skills = [
        skill for skill in required_skills
        if skill not in user_skill_list
    ]

    return jsonify({
        "target_role": target_role,
        "your_skills": user_skill_list,
        "required_skills": required_skills,
        "missing_skills": missing_skills,
        "status": "ready" if not missing_skills else "skill gap found"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)