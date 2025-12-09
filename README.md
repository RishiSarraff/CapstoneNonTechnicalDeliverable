1. Install Python 3.13.3
2. Host a Venv --> source venv/bin/active (MAC) or venv\Scripts\activate (Windows)
2. Run: pip install -r requirements.txt
3. Run: streamlit run app.py
4. Browser opens automatically to localhost:8501

## Required Data Files

The following large files are not included in this repository. Please download them separately:

Required Files:
- `AIDifficultyClassifier/ai_difficulty_system.pkl` - [Download link]
- `SimilarityScorePredictor/job_descriptions_with_embeddings.pkl` - [Download link]
- `SimilarityScorePredictor/coding_problems_with_embeddings.pkl` - [Download link]
- `Datasets/job_problem_similarity.csv` - [Download link]

Place these files in their respective directories before running the app.