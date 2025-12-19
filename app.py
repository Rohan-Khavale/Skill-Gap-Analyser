from flask import Flask, request, jsonify, send_from_directory
import json

# ----------------------------
# Load configuration files
# ----------------------------
with open("skills_master.json") as f:
    SKILLS_MASTER = json.load(f)

with open("role_skill_map.json") as f:
    ROLE_SKILL_MAP = json.load(f)

app = Flask(__name__, static_folder="static")


# ----------------------------
# Core logic
# ----------------------------
def get_skills_for_role(role: str):
    role = role.lower().strip()

    if role not in ROLE_SKILL_MAP:
        return []

    config = ROLE_SKILL_MAP[role]
    skills = []

    for domain in config.get("primary_domains", []):
        skills.extend(SKILLS_MASTER.get(domain, []))

    for domain in config.get("secondary_domains", []):
        skills.extend(SKILLS_MASTER.get(domain, []))

    return sorted(list(set(skills)))


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/healthz")
def healthz():
    return "OK", 200


@app.route("/skills/<role>", methods=["GET"])
def skills_for_role(role):
    return jsonify({
        "skills": get_skills_for_role(role)
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    user_skills = data.get("skills", "").lower()
    target_role = data.get("target", "").lower()

    required_skills = get_skills_for_role(target_role)

    if not required_skills:
        return jsonify({
            "error": "Target role not supported",
            "supported_roles": list(ROLE_SKILL_MAP.keys())
        })

    user_skill_list = [s.strip() for s in user_skills.split(",") if s.strip()]

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


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=True)