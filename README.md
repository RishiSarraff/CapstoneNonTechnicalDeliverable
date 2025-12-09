## Open StreamLit using Deployed app Link:
Link: https://aijoblistings.streamlit.app/ (.pkl and .csv files might be too large, if so, run locally)

# Setup Instructions

## Prerequisites
- Python 3.8 or higher
- API keys (provided in submission)

---

## Installation Steps

1. **Clone repository**
   ```bash
   git clone https://github.com/RishiSarraff/CapstoneNonTechnicalDeliverable.git
   cd CapstoneNonTechnicalDeliverable
   ```

2. **Create `.env` file** in project root with:
   ```
   DEEPSEEK_API_KEY=your-deepseek-key-here(in project submission)
   ```

3. **Set up virtual environment**
   
   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   Windows:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run application**
   ```bash
   streamlit run app.py
   ```

App opens at: **http://localhost:8501**

---

## Quick Troubleshooting
- **API key error:** Check `.env` file exists with both keys
- **Module error:** Activate venv, run `pip install -r requirements.txt`
- **Port in use:** Add `--server.port 8502` to run command

## Required Data Files

The following large files are not included in this repository. Please download them separately:

Required Files:
- `AIDifficultyClassifier/ai_difficulty_system.pkl`
- `SimilarityScorePredictor/job_descriptions_with_embeddings.pkl`
- `SimilarityScorePredictor/coding_problems_with_embeddings.pkl`
- `Datasets/job_problem_similarity.csv`

Place these files in their respective directories before running the app.