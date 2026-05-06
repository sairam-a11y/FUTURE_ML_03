"""
rank_candidates.py
------------------
Core scoring engine:
  - TF-IDF vectorisation
  - Cosine similarity against job description
  - Weighted skill-overlap scoring
  - Hybrid score computation
  - Skill gap analysis
  - Ranking with full explainability
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_text, remove_stopwords, extract_skills, SKILL_DICT


# ── Weight configuration ─────────────────────────────────────────────────────
WEIGHTS = {
    "tfidf_similarity": 0.45,   # semantic text match
    "skill_overlap":    0.40,   # hard skill coverage
    "resume_richness":  0.15,   # bonus for detailed resume
}


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def _to_text(value) -> str:
    """Coerce any missing or non-string value into a safe text string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def extract_jd_skills(job_description: str) -> list:
    """Extract required skills from a job description string."""
    return extract_skills(job_description)


def build_tfidf_scores(
    processed_df: pd.DataFrame,
    job_description: str,
    text_col: str = "filtered_text",
) -> np.ndarray:
    """
    Fit TF-IDF on all resume text + JD, then return cosine similarity
    of each resume against the JD.
    """
    clean_jd = _to_text(remove_stopwords(clean_text(job_description)))
    resume_texts = [_to_text(text) for text in processed_df[text_col].tolist()]
    corpus = resume_texts + [clean_jd]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vecs = tfidf_matrix[:-1]          # all resumes
    jd_vec = tfidf_matrix[-1]                # job description

    scores = cosine_similarity(resume_vecs, jd_vec).flatten()
    return scores, vectorizer


def compute_skill_overlap(
    processed_df: pd.DataFrame,
    jd_skills: list,
) -> pd.Series:
    """
    Jaccard-like overlap: (matched skills) / (total required skills).
    Returns values in [0, 1].
    """
    if not jd_skills:
        return pd.Series(0.5, index=processed_df.index)

    def overlap(skills_list):
        if not skills_list:
            return 0.0
        matched = len(set(skills_list) & set(jd_skills))
        return matched / len(jd_skills)

    return processed_df["skills"].apply(overlap)


def compute_richness(processed_df: pd.DataFrame) -> pd.Series:
    """
    Resume richness = normalised word count (rewards detailed resumes).
    """
    return _normalize(processed_df["word_count"].clip(upper=1500))


def score_and_rank(
    processed_df: pd.DataFrame,
    job_description: str,
    top_k: int = None,
) -> pd.DataFrame:
    """
    Full scoring pipeline.

    Parameters
    ----------
    processed_df : preprocessed resume DataFrame (from preprocess.py)
    job_description : raw JD string entered by user
    top_k : if set, return only top-k candidates

    Returns
    -------
    ranked DataFrame with scores and skill gap columns
    """
    df = processed_df.copy()

    # 1. TF-IDF cosine similarity
    tfidf_raw, vectorizer = build_tfidf_scores(df, job_description)
    df["tfidf_score"] = tfidf_raw

    # 2. JD skill extraction & overlap
    jd_skills = extract_jd_skills(job_description)
    df["skill_overlap_raw"] = compute_skill_overlap(df, jd_skills)
    df["skill_overlap"] = df["skill_overlap_raw"]   # already in [0,1]

    # 3. Resume richness
    df["richness"] = compute_richness(df)

    # 4. Normalize tfidf to [0,1]
    df["tfidf_norm"] = _normalize(df["tfidf_score"])

    # 5. Weighted hybrid score (0–100)
    df["final_score"] = (
        WEIGHTS["tfidf_similarity"]  * df["tfidf_norm"]
        + WEIGHTS["skill_overlap"]   * df["skill_overlap"]
        + WEIGHTS["resume_richness"] * df["richness"]
    ) * 100

    # 6. Skill gap analysis
    df["matched_skills"] = df["skills"].apply(
        lambda s: sorted(set(s) & set(jd_skills))
    )
    df["missing_skills"] = df["skills"].apply(
        lambda s: sorted(set(jd_skills) - set(s))
    )
    df["extra_skills"] = df["skills"].apply(
        lambda s: sorted(set(s) - set(jd_skills))
    )
    df["match_count"]   = df["matched_skills"].apply(len)
    df["missing_count"] = df["missing_skills"].apply(len)

    # 7. Rank
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    if top_k:
        df = df.head(top_k)

    return df, jd_skills, vectorizer


def build_score_explanation(row: pd.Series) -> str:
    """Return a human-readable score breakdown for a single candidate row."""
    lines = [
        f"Hybrid Score : {row['final_score']:.1f}/100",
        f"  TF-IDF similarity : {row['tfidf_norm']*100:.1f}  (weight {WEIGHTS['tfidf_similarity']*100:.0f}%)",
        f"  Skill overlap      : {row['skill_overlap']*100:.1f}  (weight {WEIGHTS['skill_overlap']*100:.0f}%)",
        f"  Resume richness    : {row['richness']*100:.1f}  (weight {WEIGHTS['resume_richness']*100:.0f}%)",
        f"  Matched skills     : {', '.join(row['matched_skills']) or 'none'}",
        f"  Missing skills     : {', '.join(row['missing_skills']) or 'none'}",
    ]
    return "\n".join(lines)


def evaluate_ranking(ranked_df: pd.DataFrame, jd_skills: list) -> dict:
    """
    Compute evaluation metrics:
    - Score distribution statistics
    - Precision@K (fraction of top-K with >50 score)
    - Skill coverage @ various K values
    """
    scores = ranked_df["final_score"]
    metrics = {
        "total_candidates": len(ranked_df),
        "score_mean": round(float(scores.mean()), 2),
        "score_std":  round(float(scores.std()), 2),
        "score_min":  round(float(scores.min()), 2),
        "score_max":  round(float(scores.max()), 2),
        "score_median": round(float(scores.median()), 2),
        "required_skills_count": len(jd_skills),
        "required_skills": jd_skills,
        "weights_used": WEIGHTS,
    }

    for k in [5, 10, 20]:
        if len(ranked_df) >= k:
            top_k = ranked_df.head(k)
            prec = (top_k["final_score"] > 50).mean()
            metrics[f"precision_at_{k}"] = round(float(prec), 3)
            avg_match = top_k["match_count"].mean()
            metrics[f"avg_skill_match_top_{k}"] = round(float(avg_match), 2)

    return metrics


def save_outputs(
    ranked_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    jd_skills: list,
    metrics: dict,
    output_dir: str = "../outputs",
):
    """Save all four output artefacts."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ranked_candidates.csv
    save_cols = [
        c for c in [
            "ID", "Category", "final_score", "tfidf_norm", "skill_overlap",
            "richness", "match_count", "missing_count",
            "matched_skills", "missing_skills", "extra_skills",
            "skills_str", "word_count",
        ] if c in ranked_df.reset_index().columns
    ]
    ranked_df.reset_index().to_csv(
        f"{output_dir}/ranked_candidates.csv", index=False
    )

    # extracted_skills.csv
    skills_df = processed_df[["ID", "Category", "skills_str", "skill_count"]].copy() \
        if "ID" in processed_df.columns else processed_df[["Category", "skills_str", "skill_count"]].copy()
    skills_df.to_csv(f"{output_dir}/extracted_skills.csv", index=False)

    # processed_resumes.csv (without HTML to keep file small)
    proc_save = processed_df.drop(
        columns=[c for c in ["Resume_html", "resume_text"] if c in processed_df.columns],
        errors="ignore"
    )
    proc_save.to_csv(f"{output_dir}/processed_resumes.csv", index=False)

    # metrics.json
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[✓] Outputs saved to {output_dir}/")
