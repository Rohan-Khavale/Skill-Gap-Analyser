from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT = Path("data/raw/job_descriptions")
OUT.mkdir(parents=True, exist_ok=True)

print("Downloading job descriptions dataset...")
ds = load_dataset("lukebarousse/data_jobs", split="train")

df = pd.DataFrame(ds)

df = df.rename(columns={
    "job_title": "role_title",
    "job_description": "job_description"
})

df["canonical_role"] = df["role_title"].str.lower()
df["company"] = "unknown"

df = df[["role_title", "canonical_role", "company", "job_description"]]
df = df.dropna(subset=["job_description"])

df.to_csv(OUT / "jd_real.csv", index=False)
print("Saved:", OUT / "jd_real.csv")

