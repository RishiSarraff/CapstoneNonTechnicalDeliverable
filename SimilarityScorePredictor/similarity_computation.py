import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI
import dotenv

load_dotenv()

df = pd.read_csv('jobDescriptionsDataset.csv')
df.head()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=deepseek_api_key)

def get_embedding(text, model="text-embedding-3-large"):
    if pd.isna(text) or not isinstance(text, str):
        return None
    response = client.embeddings.create(
        input=text.strip(),
        model=model
    )
    return response.data[0].embedding

# Generate embeddings
tqdm.pandas()
df["embedding"] = df["description"].progress_apply(get_embedding)

df.to_pickle("job_descriptions_with_embeddings.pkl")

df = pd.read_pickle("job_descriptions_with_embeddings.pkl")

# Convert embeddings to a numpy array for later cosine similarity
emb_matrix = np.vstack(df["embedding"].dropna())

df2 = pd.read_csv('dataset_classified.csv')
df2.head()

# Generate embeddings
tqdm.pandas()
df2["embedding"] = df2["Question Text"].progress_apply(get_embedding)

# Save to file for future use
df2.to_pickle("coding_problems_with_embeddings.pkl")

df2 = pd.read_pickle("coding_problems_with_embeddings.pkl")

df2['Platform'] = (
    ['LeetCode'] * 200 +
    ['Codeforces'] * 200 +
    ['HackerRank'] * 69
)

# Convert embeddings to a numpy array for later cosine similarity
emb_matrix2 = np.vstack(df2["embedding"].dropna())

print("len(df2):", len(df2))
print("len(emb_matrix2):", len(emb_matrix2))
print("Null embeddings in df2:", df2["embedding"].isna().sum())

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

TOP_N = 5     # top N coding problems per job
BATCH_SIZE = 50  # batch size for job descriptions

results = []

print("Computing cosine similarities in batches...")

num_jobs = len(emb_matrix)
num_problems = len(emb_matrix2)

for start in range(0, num_jobs, BATCH_SIZE):
    end = min(start + BATCH_SIZE, num_jobs)
    job_batch = emb_matrix[start:end]

    # Compute similarity for this batch
    sims = cosine_similarity(job_batch, emb_matrix2)

    # Top N per job
    top_indices = np.argsort(-sims, axis=1)[:, :TOP_N]
    top_scores = np.take_along_axis(sims, top_indices, axis=1)

    for i, job_idx in enumerate(range(start, end)):
        for j in range(TOP_N):
            problem_idx = int(top_indices[i, j])

            # --- Safety checks ---
            if problem_idx >= num_problems or job_idx >= num_jobs:
                continue

            try:
                results.append({
                    "description": df.iloc[job_idx]["description"],
                    "job_title": df.iloc[job_idx]["job_title"],
                    "job_type": df.iloc[job_idx]["job_type"],
                    "job_level": df.iloc[job_idx]["job level"],
                    "category": df.iloc[job_idx]["category"],
                    "question_text": df2.iloc[problem_idx]["Question Text"],
                    "difficulty_level": df2.iloc[problem_idx]["Difficulty Level"],
                    "platform": df2.iloc[problem_idx]["Platform"],
                    "similarity": float(top_scores[i, j])
                })
            except Exception as e:
                print(f"⚠️ Skipping pair (job {job_idx}, problem {problem_idx}): {e}")

similarity_df = pd.DataFrame(results)

similarity_df.head()

# Per-job normalization (within each job description)
similarity_df["similarity_norm_job"] = (
    similarity_df.groupby("description")["similarity"]
    .transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
)

# Per-category normalization (within each category)
similarity_df["similarity_norm_category"] = (
    similarity_df.groupby("category")["similarity"]
    .transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
)

# Save file for future use
similarity_df.to_csv("job_problem_similarity.csv", index=False)
print("Saved similarity results to job_problem_similarity.csv")

sns.histplot(similarity_df["similarity"], bins=40, kde=True)
plt.title("Distribution of Raw Cosine Similarity Scores")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.show()

sns.histplot(similarity_df["similarity_norm_job"], bins=40, kde=True)
plt.title("Distribution of Normalized Cosine Similarity Scores (Grouped by JD)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.show()

sns.histplot(similarity_df["similarity_norm_category"], bins=40, kde=True)
plt.title("Distribution of Normalized Cosine Similarity Scores (Grouped by Category)")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.show()

sns.boxplot(
    data=similarity_df,
    x="category",
    y="similarity"
)
plt.title("Similarity Distribution by Job Category")
plt.xticks(rotation=45)
plt.show()

BATCH_SIZE = 50  # number of job descriptions processed per batch
results = []

num_jobs = len(emb_matrix)
num_problems = len(emb_matrix2)

print(f"Computing ALL cosine similarities for {num_jobs} jobs × {num_problems} problems...")

# ---- Precompute category-level stats ----
# We'll store all similarities temporarily by category to compute mean/std later
category_sims = {cat: [] for cat in df["category"].unique()}

# Outer progress bar for batches
for start in tqdm(range(0, num_jobs, BATCH_SIZE), desc="Processing job batches"):
    end = min(start + BATCH_SIZE, num_jobs)
    job_batch = emb_matrix[start:end]

    # Compute similarities for this batch (shape: batch_size × num_problems)
    sims = cosine_similarity(job_batch, emb_matrix2)

    # Inner loop over each job in this batch
    for i, job_idx in enumerate(range(start, end)):
        job_row = df.iloc[job_idx]
        job_sims = sims[i]

        # Save similarities for category-level normalization later
        category_sims[job_row["category"]].extend(job_sims.tolist())

        # Compute per-job normalization (z-score)
        mean_job = job_sims.mean()
        std_job = job_sims.std() if job_sims.std() != 0 else 1
        normalized_job = (job_sims - mean_job) / std_job

        # Store all job–problem pairs for this job
        for problem_idx in range(num_problems):
            problem_row = df2.iloc[problem_idx]
            results.append({
                "description": job_row["description"],
                "job_title": job_row["job_title"],
                "job_type": job_row["job_type"],
                "job_level": job_row["job level"],
                "category": job_row["category"],
                "question_text": problem_row["Question Text"],
                "difficulty_level": problem_row["Difficulty Level"],
                "platform": problem_row["Platform"],
                "similarity": float(job_sims[problem_idx]),
                "similarity_norm_job": float(normalized_job[problem_idx]),
                # We'll fill similarity_norm_category later
                "similarity_norm_category": None
            })

print(f"Step 1/2 Computed raw + per-job normalized similarities for {len(results)} records.")

# ---- Compute category-level means and stds ----
category_stats = {}
for cat, sims in category_sims.items():
    sims = np.array(sims)
    mean_cat = sims.mean()
    std_cat = sims.std() if sims.std() != 0 else 1
    category_stats[cat] = (mean_cat, std_cat)

print("Step 2/2 Computed category-level means and standard deviations.")

# ---- Fill in per-category normalization ----
for r in tqdm(results, desc="Applying per-category normalization"):
    cat = r["category"]
    mean_cat, std_cat = category_stats[cat]
    r["similarity_norm_category"] = (r["similarity"] - mean_cat) / std_cat

# --- Save results to JSON file ---
output_path = "job_problem_similarity_all.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved all similarities with per-job and per-category normalization to {output_path}")
