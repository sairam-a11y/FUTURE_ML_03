# 🎯 Resume Screening & Candidate Ranking System

An end-to-end ML hiring assistant that reads real resume text, extracts skills, scores candidates against a job description, ranks them, and highlights skill gaps — built entirely on the attached **Resume.csv** dataset.

---

## 📁 Project Structure

```
resume-screening-project/
├── data/
│   └── resumes.csv              # Source dataset (2,484 real resumes, 24 categories)
├── src/
│   ├── preprocess.py            # Text cleaning, skill extraction, feature engineering
│   ├── rank_candidates.py       # TF-IDF scoring, hybrid ranking, skill gap analysis
│   └── visualize.py             # All 8 business-quality charts
├── outputs/
│   ├── ranked_candidates.csv    # All candidates with scores & skill gaps
│   ├── extracted_skills.csv     # Per-resume skill extraction
│   ├── processed_resumes.csv    # Cleaned/featurised resumes
│   ├── metrics.json             # Evaluation metrics
│   └── *.png                    # 8 visualisation charts
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Used

**File:** `Resume.csv`  
**Rows:** 2,484 real resumes  
**Columns auto-detected:**

| Column | Role |
|--------|------|
| `ID` | Candidate identifier |
| `Resume_str` | Raw resume text → used for all NLP |
| `Resume_html` | HTML version (ignored after text extraction) |
| `Category` | Job role label (24 categories) |

**Categories include:** INFORMATION-TECHNOLOGY, BUSINESS-DEVELOPMENT, HR, FINANCE, ENGINEERING, HEALTHCARE, BANKING, SALES, CONSULTANT, and 15 more.

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the batch pipeline (outputs to /outputs/)
```bash
cd resume-screening-project
python3 -c "
import sys; sys.path.insert(0, 'src')
from preprocess import preprocess_dataframe
from rank_candidates import score_and_rank, evaluate_ranking, save_outputs
from visualize import generate_all_charts
import pandas as pd

raw = pd.read_csv('data/resumes.csv')
processed = preprocess_dataframe(raw)

JD = 'Your job description here with required skills...'
ranked, jd_skills, _ = score_and_rank(processed, JD)
metrics = evaluate_ranking(ranked, jd_skills)
save_outputs(ranked, processed, jd_skills, metrics, 'outputs')
generate_all_charts(processed, ranked, jd_skills, 'outputs')
"
```

### 3. Launch the interactive dashboard
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🧠 How Scoring Works

Each candidate receives a **Hybrid Score (0–100)** calculated as:

```
Hybrid Score = (TF-IDF Similarity × 0.45)
             + (Skill Overlap × 0.40)
             + (Resume Richness × 0.15)
             × 100
```

### Component breakdown

| Component | Weight | Method |
|-----------|--------|--------|
| **TF-IDF Similarity** | 45% | Cosine similarity between resume TF-IDF vector and JD TF-IDF vector (bigrams, 8k features, sublinear TF) |
| **Skill Overlap** | 40% | `matched_skills / total_jd_skills` — hard skill coverage ratio |
| **Resume Richness** | 15% | Min-max normalised word count (rewards detailed, well-written resumes) |

All three components are normalised to [0,1] before weighting.

---

## 🏆 Why Candidates Rank Higher

A candidate ranks **higher** when they:
1. Have **more JD-required skills** in their resume (highest impact, 40% weight)
2. Write about **similar topics** as the job description — their resume text is semantically closer (45% weight)
3. Have a **detailed, comprehensive resume** with rich content (15% bonus)

A candidate ranks **lower** when they:
- Are missing most required technical skills
- Work in an unrelated domain (low TF-IDF similarity)
- Have a very thin resume with little content

---

## 🔍 Skill Gap Explanation

For every candidate the system produces:

| Field | Description |
|-------|-------------|
| `matched_skills` | Skills the candidate HAS that the JD requires |
| `missing_skills` | Skills the JD requires that the candidate LACKS |
| `extra_skills` | Skills the candidate has beyond JD requirements |
| `match_count` | Count of matched skills |
| `missing_count` | Count of missing skills |

Skills are extracted using a dictionary of **200+ domain skills** (programming languages, ML frameworks, cloud tools, soft skills, finance, HR, healthcare terms) matched via regex against the full resume text.

---

## 📈 Evaluation Metrics

Since this is an unsupervised ranking problem (no ground-truth relevance labels), we evaluate using:

- **Precision@K** — fraction of top-K candidates scoring above 50 (threshold for "relevant")
- **Score distribution** — mean, std, min/max, median
- **Average skill match @K** — mean matched skills in top-K candidates
- **Consistency checks** — top candidates always have more matched skills than bottom candidates

Sample results for a Data Scientist JD:
- Precision@5 = 1.0 (all top-5 scored >50)
- Avg skill match top-10 = 5.2 skills
- Score range: 0–69 (well-distributed)

---

## 📊 Visualisations Generated

| Chart | File |
|-------|------|
| Top 20 skills overall | `top_skills.png` |
| Category distribution | `category_dist.png` |
| Resume word-count distribution | `resume_lengths.png` |
| Candidate score distribution | `score_dist.png` |
| Ranking leaderboard (top 15) | `leaderboard.png` |
| Skill match heatmap | `skill_heatmap.png` |
| Most frequently missing skills | `missing_skills.png` |
| Role-wise skill demand | `role_skill_demand.png` |

---

## 💼 Business Value for Hiring Teams

| Pain Point | How This System Helps |
|------------|----------------------|
| Manual screening of 100s of CVs | Auto-ranks all 2,484 in <10 seconds |
| Inconsistent shortlisting | Objective, reproducible scoring formula |
| Missing skill visibility | Explicit missing-skills list per candidate |
| Recruiter bias | Blind scoring on text + skills only |
| JD-candidate mismatch | Semantic + skill matching catches both exact and related matches |
| Reporting to management | Downloadable ranked CSV + metrics JSON |

---

## 🛠 Tech Stack

- **Python 3.9+** — core language
- **pandas / numpy** — data manipulation
- **scikit-learn** — TF-IDF vectorisation, cosine similarity
- **matplotlib** — all visualisations
- **Streamlit** — interactive web dashboard
- No NLTK/spaCy required — fully self-contained

---

## 📌 Notes

- The `Resume_html` column is stripped and ignored; all NLP uses `Resume_str`
- Stop words use sklearn's built-in English set extended with resume-specific filler words
- The skill dictionary covers 200+ skills across IT, Finance, HR, Healthcare, Engineering, Design, and Business domains
- Weights are adjustable in the Streamlit sidebar without restarting
