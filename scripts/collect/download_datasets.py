from datasets import load_dataset
import pandas as pd
from pathlib import Path

BASE = Path("data/raw")

# ------------------------
# Job Descriptions
# ------------------------
print("Downloading job descriptions...")
jd = load_dataset("jacob-huggingface/job_descriptions", split="train")
jd_df = pd.DataFrame(jd)

jd_df = jd_df.rename(columns={
    "job_title": "role_title",
    "job_description": "job_description"
})

jd_df["canonical_role"] = jd_df["role_title"].str.lower()
jd_df["company"] = "unknown"
jd_df["seniority"] = "unknown"
jd_df["location"] = "unknown"

jd_df.insert(0, "job_id", range(1, len(jd_df)+1))

jd_df[
    ["job_id","role_title","canonical_role","company","seniority","location","job_description"]
].to_csv(BASE / "job_descriptions" / "jd_hf.csv", index=False)

# ------------------------
# Resumes
# ------------------------
print("Downloading resumes...")
res = load_dataset("Hiring/ResumeDataset", split="train")
res_df = pd.DataFrame(res)

res_df = res_df.rename(columns={"Resume_str": "resume_text"})
res_df["current_role"] = "unknown"
res_df["target_role"] = "unknown"
res_df.insert(0, "candidate_id", range(1, len(res_df)+1))

res_df[
    ["candidate_id","resume_text","current_role","target_role"]
].to_csv(BASE / "resumes" / "resumes_hf.csv", index=False)

print("Done.")
