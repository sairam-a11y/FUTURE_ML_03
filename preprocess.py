"""
preprocess.py
-------------
Text cleaning, normalization, and feature extraction for resume data.
Uses sklearn's built-in stop words (no NLTK dependency required).
"""

import re
import string
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Extended stop words combining sklearn's set with common resume filler words
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "company", "city", "state", "current", "name", "work", "worked",
    "experience", "skills", "education", "summary", "responsibilities",
    "able", "also", "including", "various", "ensure", "new", "using",
    "use", "used", "well", "strong", "excellent", "good", "great",
    "responsible", "related", "based", "role", "job", "position",
    "years", "year", "month", "months", "team", "company", "organization",
}


# ── Skill dictionary ────────────────────────────────────────────────────────

SKILL_DICT = {
    # Programming & Dev
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "ruby",
    "php", "scala", "kotlin", "swift", "r", "matlab", "sql", "nosql", "bash",
    "shell", "perl", "rust",
    # Web & Frameworks
    "react", "angular", "vue", "nodejs", "django", "flask", "fastapi", "spring",
    "html", "css", "bootstrap", "jquery", "rest", "graphql", "api",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "ci/cd", "devops", "linux", "unix", "git", "github", "gitlab", "ansible",
    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "spark",
    "hadoop", "kafka", "airflow", "tableau", "power bi", "excel", "statistics",
    "data analysis", "data science", "big data", "etl", "data engineering",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "cassandra", "oracle", "sqlite",
    "elasticsearch", "dynamodb",
    # Business & Soft
    "project management", "agile", "scrum", "leadership", "communication",
    "problem solving", "strategic planning", "budgeting", "forecasting",
    "marketing", "sales", "crm", "customer service", "negotiation",
    "stakeholder management", "business development", "analytics",
    # Finance
    "financial modeling", "accounting", "auditing", "tax", "quickbooks",
    "sap", "erp", "risk management", "compliance", "investment", "banking",
    # HR
    "recruitment", "talent acquisition", "onboarding", "payroll", "hris",
    "employee relations", "performance management", "training", "hr",
    # Design
    "photoshop", "illustrator", "figma", "sketch", "adobe xd", "indesign",
    "ui/ux", "wireframing", "prototyping",
    # Healthcare
    "patient care", "clinical", "nursing", "medical", "ehr", "hipaa",
    "pharmacology", "diagnosis",
    # Engineering
    "autocad", "solidworks", "catia", "matlab", "simulation", "cad", "cam",
    "mechanical", "electrical", "civil", "structural", "quality control",
}


def clean_text(text: str) -> str:
    """
    Normalize resume text:
    - Lowercase
    - Remove HTML tags
    - Remove special characters / extra whitespace
    - Remove digits-only tokens
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove punctuation (keep hyphens inside words like c++)
    text = re.sub(r"[^a-z0-9\s\+\#\/]", " ", text)
    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove stop words from cleaned text."""
    tokens = text.split()
    return " ".join(t for t in tokens if t not in STOP_WORDS and len(t) > 1)


def extract_skills(text: str) -> list:
    """
    Match skills from the predefined SKILL_DICT against resume text.
    Handles both single-word and multi-word skill phrases.
    """
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    found = []
    for skill in SKILL_DICT:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(found)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: detect columns, clean text, extract skills, add features.

    Parameters
    ----------
    df : raw DataFrame loaded from resumes.csv

    Returns
    -------
    pd.DataFrame with added columns:
        clean_text, filtered_text, skills, skill_count, word_count
    """
    df = df.copy()

    # ── Auto-detect columns ─────────────────────────────────────────────────
    col_lower = {c.lower(): c for c in df.columns}

    resume_col = None
    for candidate in ["resume_str", "resume", "text", "cv", "profile", "content"]:
        if candidate in col_lower:
            resume_col = col_lower[candidate]
            break
    if resume_col is None:
        # Fallback: pick the longest text column
        text_cols = df.select_dtypes(include="object").columns
        resume_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())

    id_col = None
    for candidate in ["id", "candidate_id", "applicant_id"]:
        if candidate in col_lower:
            id_col = col_lower[candidate]
            break

    category_col = None
    for candidate in ["category", "role", "job_role", "position", "title"]:
        if candidate in col_lower:
            category_col = col_lower[candidate]
            break

    # Standardise column names
    df.rename(columns={resume_col: "resume_text"}, inplace=True)
    if id_col and id_col != "ID":
        df.rename(columns={id_col: "ID"}, inplace=True)
    if category_col and category_col != "Category":
        df.rename(columns={category_col: "Category"}, inplace=True)

    # ── Drop duplicates & nulls ─────────────────────────────────────────────
    df.drop_duplicates(subset=["resume_text"], inplace=True)
    df.dropna(subset=["resume_text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Text cleaning ───────────────────────────────────────────────────────
    df["clean_text"] = df["resume_text"].apply(clean_text)
    df["filtered_text"] = df["clean_text"].apply(remove_stopwords)

    # ── Feature engineering ─────────────────────────────────────────────────
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["skills"] = df["resume_text"].apply(extract_skills)
    df["skill_count"] = df["skills"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda s: ", ".join(s))

    return df


def get_column_info(df: pd.DataFrame) -> dict:
    """Return detected column mapping for logging / debugging."""
    return {
        "id_col": "ID" if "ID" in df.columns else None,
        "resume_col": "resume_text",
        "category_col": "Category" if "Category" in df.columns else None,
        "total_rows": len(df),
        "columns": df.columns.tolist(),
    }
