import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Dummy training data (demo only)
texts = [
    "python machine learning data science",
    "deep learning neural networks",
    "docker kubernetes devops aws",
    "linux bash scripting monitoring",
]
labels = ["ML", "ML", "DevOps", "DevOps"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(texts, labels)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/demo_pipeline.joblib")

print("✅ Model saved to models/demo_pipeline.joblib")
