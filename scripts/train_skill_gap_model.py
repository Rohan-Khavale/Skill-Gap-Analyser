import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path("data/raw/job_descriptions")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading job descriptions...")
df = pd.read_csv(BASE / "jd_real.csv")

df["job_description"] = df["job_description"].fillna("").str.lower()
df["canonical_role"] = df["canonical_role"].fillna("").str.lower()

print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["job_description"])

joblib.dump(vectorizer, MODEL_DIR / "skill_vectorizer.joblib")
joblib.dump(X, MODEL_DIR / "role_skill_matrix.joblib")
joblib.dump(df["canonical_role"].tolist(), MODEL_DIR / "roles.joblib")

print("Model training complete.")
