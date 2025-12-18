import requests
import csv
from pathlib import Path

OUT = Path("data/raw/job_descriptions")
OUT.mkdir(parents=True, exist_ok=True)

roles = [
    "software engineer",
    "data scientist",
    "backend developer",
    "frontend developer",
    "devops engineer",
    "machine learning engineer",
    "data analyst",
    "cloud engineer"
]

headers = ["job_id", "role_title", "canonical_role", "company", "job_description"]

with open(OUT / "jd_real.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    job_id = 0
    for role in roles:
        url = f"https://api.adzuna.com/v1/api/jobs/gb/search/1"
        params = {
            "app_id": "demo",
            "app_key": "demo",
            "what": role
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                continue

            data = r.json()
            for j in data.get("results", []):
                writer.writerow([
                    job_id,
                    j.get("title", ""),
                    role,
                    j.get("company", {}).get("display_name", ""),
                    j.get("description", "")
                ])
                job_id += 1
        except Exception:
            continue

print("Job descriptions saved.")
