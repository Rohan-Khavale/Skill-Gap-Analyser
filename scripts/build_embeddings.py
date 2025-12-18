import pandas as pd
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading datasets...")

roles = pd.read_csv(BASE / "roles" / "role_aliases.csv")
skills = pd.read_csv(BASE / "skills" / "skill_aliases.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating role embeddings...")
role_texts = roles["canonical_role"].astype(str).unique().tolist()
role_embeddings = model.encode(role_texts, show_progress_bar=True)

print("Generating skill embeddings...")
skill_texts = skills["canonical_skill"].astype(str).unique().tolist()
skill_embeddings = model.encode(skill_texts, show_progress_bar=True)

joblib.dump(role_texts, MODEL_DIR / "role_labels.joblib")
joblib.dump(role_embeddings, MODEL_DIR / "role_embeddings.joblib")

joblib.dump(skill_texts, MODEL_DIR / "skill_labels.joblib")
joblib.dump(skill_embeddings, MODEL_DIR / "skill_embeddings.joblib")

print("Embeddings saved to /models")
