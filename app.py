"""
app.py
------
Streamlit dashboard for the Resume Screening & Ranking system.

Run:
    streamlit run app.py

Features:
- Upload any CSV with resume text
- Paste a job description
- Auto-preprocess and score all candidates
- Leaderboard, charts, skill gap report
- Download ranked results
"""

import ast
import json
import os
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from preprocess import clean_text, extract_skills, preprocess_dataframe, remove_stopwords
from rank_candidates import evaluate_ranking, save_outputs, score_and_rank
from visualize import (
    plot_category_distribution,
    plot_leaderboard,
    plot_missing_skills,
    plot_role_skill_demand,
    plot_score_distribution,
    plot_skill_heatmap,
    plot_top_skills,
    plot_resume_length,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "resumes.csv")
FALLBACK_PATH = os.path.join(BASE_DIR, "ranked_candidates.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(
    page_title="Resume Screener | AI Hiring Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _format_number(value) -> str:
    if value is None:
        return "—"
    try:
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return f"{float(value):.1f}"
    except Exception:
        return str(value)


def _parse_list_value(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is not None and not isinstance(parsed, (str, bytes)):
        parsed_text = str(parsed).strip()
        return [parsed_text] if parsed_text else []
    return [item.strip() for item in text.split(",") if item.strip()]


def _prepare_fallback_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    if "resume_text" not in df.columns:
        for candidate in ["Resume_str", "resume", "text", "description", "content"]:
            if candidate in df.columns:
                rename_map[candidate] = "resume_text"
                break
    if "ID" not in df.columns:
        for candidate in ["id", "candidate_id", "applicant_id", "Rank"]:
            if candidate in df.columns:
                rename_map[candidate] = "ID"
                break
    if "Category" not in df.columns:
        for candidate in ["category", "role", "job_role", "title"]:
            if candidate in df.columns:
                rename_map[candidate] = "Category"
                break

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    if "resume_text" not in df.columns:
        text_cols = df.select_dtypes(include="object").columns.tolist()
        if text_cols:
            df["resume_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)
        else:
            df["resume_text"] = ""

    if "Category" not in df.columns:
        df["Category"] = "Unknown"
    if "ID" not in df.columns:
        df["ID"] = np.arange(1, len(df) + 1)

    if "Rank" in df.columns:
        df.drop(columns=["Rank"], inplace=True)

    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    df["clean_text"] = df["resume_text"].apply(clean_text)
    df["filtered_text"] = df["clean_text"].apply(remove_stopwords)
    df["word_count"] = df["clean_text"].apply(lambda text: len(str(text).split()))

    if "skills" in df.columns:
        parsed_skills = df["skills"].apply(_parse_list_value)
        df["skills"] = parsed_skills.apply(lambda skills: skills if skills else extract_skills("") )
    else:
        df["skills"] = df["resume_text"].apply(extract_skills)

    df["skill_count"] = df["skills"].apply(len)
    df["skills_str"] = df["skills"].apply(lambda skills: ", ".join(skills))
    return df


@st.cache_data(show_spinner="Loading and preparing the resume corpus…")
def load_and_preprocess(source: str = "auto", file_bytes=None):
    import io

    meta = {
        "source": source,
        "label": "",
        "note": "",
    }

    if source == "upload" and file_bytes is not None:
        raw = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        processed = preprocess_dataframe(raw)
        meta.update(
            {
                "source": "upload",
                "label": "Uploaded CSV",
                "note": "Your uploaded file was cleaned and scored on the fly.",
            }
        )
        return processed, meta

    prefer_original = source in {"auto", "builtin"}
    prefer_fallback = source in {"auto", "fallback"}

    if prefer_original and os.path.exists(DATA_PATH):
        raw = pd.read_csv(DATA_PATH, low_memory=False)
        processed = preprocess_dataframe(raw)
        meta.update(
            {
                "source": "builtin",
                "label": "Original resumes.csv",
                "note": "Loaded the full source resume dataset.",
            }
        )
        return processed, meta

    if prefer_fallback and os.path.exists(FALLBACK_PATH):
        fallback_cols = [
            "Rank",
            "ID",
            "resume_text",
            "Resume_html",
            "Category",
            "clean_text",
            "filtered_text",
            "word_count",
            "skills",
            "skill_count",
            "skills_str",
        ]
        preview_rows = 650 if source == "auto" else None
        fallback_df = pd.read_csv(
            FALLBACK_PATH,
            usecols=lambda column: column in fallback_cols,
            low_memory=False,
            nrows=preview_rows,
        )
        processed = _prepare_fallback_dataframe(fallback_df)
        meta.update(
            {
                "source": "fallback",
                "label": "Cached ranked_candidates.csv",
                "note": (
                    "The original resumes.csv is missing, so the dashboard uses the saved ranking corpus as a high-quality fallback."
                    if preview_rows is None
                    else "The original resumes.csv is missing, so the dashboard uses a fast preview sample from the saved ranking corpus."
                ),
            }
        )
        return processed, meta

    demo_rows = [
        {
            "ID": 1,
            "Category": "Data Science",
            "resume_text": "Python data scientist with machine learning, pandas, numpy, scikit-learn, SQL, AWS, and Power BI experience. Built dashboards, cleaned data, and deployed models.",
        },
        {
            "ID": 2,
            "Category": "Software Engineering",
            "resume_text": "Software engineer skilled in Java, Python, REST API design, Docker, Kubernetes, Git, CI/CD, and agile delivery. Strong problem solving and system design.",
        },
        {
            "ID": 3,
            "Category": "Human Resources",
            "resume_text": "HR professional focused on recruitment, talent acquisition, onboarding, payroll, employee relations, HRIS, and performance management.",
        },
        {
            "ID": 4,
            "Category": "Finance",
            "resume_text": "Finance analyst with accounting, financial modeling, forecasting, budgeting, Excel, SQL, ERP, compliance, and risk management experience.",
        },
        {
            "ID": 5,
            "Category": "Healthcare",
            "resume_text": "Healthcare specialist with patient care, clinical documentation, diagnosis support, HIPAA awareness, EHR workflows, and nursing collaboration.",
        },
    ]
    processed = preprocess_dataframe(pd.DataFrame(demo_rows))
    meta.update(
        {
            "source": "synthetic",
            "label": "Synthetic demo corpus",
            "note": "No dataset files were available, so the dashboard generated a small interactive demo corpus.",
        }
    )
    return processed, meta


def _score_band(score: float) -> str:
    if score >= 75:
        return "Elite"
    if score >= 60:
        return "Strong"
    if score >= 45:
        return "Warm"
    return "Needs review"


def _count_missing_skills(ranked_df: pd.DataFrame, limit: int = 10) -> list[tuple[str, int]]:
    counter = Counter()
    for skill_list in ranked_df.head(limit).get("missing_skills", []):
        if isinstance(skill_list, list):
            counter.update(skill_list)
    return counter.most_common(12)


def _candidate_options(ranked_df: pd.DataFrame) -> list[int]:
    if ranked_df is None or ranked_df.empty:
        return []
    return [int(rank) for rank in ranked_df.index.tolist() if pd.notna(rank)]


def _render_hero(processed_df: pd.DataFrame, source_meta: dict):
    total_rows = len(processed_df)
    category_count = processed_df["Category"].nunique() if "Category" in processed_df.columns else 0
    average_skills = processed_df["skill_count"].mean() if "skill_count" in processed_df.columns else 0
    average_words = processed_df["word_count"].mean() if "word_count" in processed_df.columns else 0

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.86));
                    border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 28px; padding: 1.4rem 1.5rem; margin-bottom: 1rem;
                    box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28); position: relative; overflow: hidden;">
            <div style="position:absolute; inset:0; background: radial-gradient(circle at top right, rgba(34, 197, 94, 0.18), transparent 28%), radial-gradient(circle at left bottom, rgba(59, 130, 246, 0.18), transparent 28%);"></div>
            <div style="position: relative; z-index: 1;">
                <div style="display:flex; flex-wrap:wrap; gap:0.45rem; margin-bottom: 0.9rem;">
                    <span style="padding:0.32rem 0.7rem; border-radius: 999px; background: rgba(59, 130, 246, 0.18); color: #DBEAFE; border: 1px solid rgba(59, 130, 246, 0.28); font-size: 0.78rem;">AI hiring cockpit</span>
                    <span style="padding:0.32rem 0.7rem; border-radius: 999px; background: rgba(16, 185, 129, 0.16); color: #D1FAE5; border: 1px solid rgba(16, 185, 129, 0.28); font-size: 0.78rem;">Hybrid scoring</span>
                    <span style="padding:0.32rem 0.7rem; border-radius: 999px; background: rgba(245, 158, 11, 0.16); color: #FEF3C7; border: 1px solid rgba(245, 158, 11, 0.28); font-size: 0.78rem;">Fallback-safe</span>
                </div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 2.55rem; line-height: 1.02; font-weight: 800; color: #F8FAFC; max-width: 14ch;">
                    Resume Screening, rebuilt as a command center.
                </div>
                <div style="font-size: 1rem; color: #CBD5E1; max-width: 960px; margin-top: 0.55rem;">
                    Upload a resume CSV, rank candidates against any job description, inspect the exact skill gaps, compare finalists side by side, and export decision-ready outputs.
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.65rem; margin-top: 1rem; color: #E2E8F0;">
                    <div style="padding:0.5rem 0.8rem; border-radius: 16px; background: rgba(15, 23, 42, 0.75); border:1px solid rgba(148, 163, 184, 0.16);">Corpus: <strong>{source_meta.get('label', 'Unknown')}</strong></div>
                    <div style="padding:0.5rem 0.8rem; border-radius: 16px; background: rgba(15, 23, 42, 0.75); border:1px solid rgba(148, 163, 184, 0.16);">Resumes: <strong>{total_rows:,}</strong></div>
                    <div style="padding:0.5rem 0.8rem; border-radius: 16px; background: rgba(15, 23, 42, 0.75); border:1px solid rgba(148, 163, 184, 0.16);">Categories: <strong>{category_count:,}</strong></div>
                    <div style="padding:0.5rem 0.8rem; border-radius: 16px; background: rgba(15, 23, 42, 0.75); border:1px solid rgba(148, 163, 184, 0.16);">Avg skills/resume: <strong>{average_skills:.1f}</strong></div>
                    <div style="padding:0.5rem 0.8rem; border-radius: 16px; background: rgba(15, 23, 42, 0.75); border:1px solid rgba(148, 163, 184, 0.16);">Avg words/resume: <strong>{average_words:.0f}</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_weights(w_tfidf: float, w_skill: float, w_rich: float):
    import rank_candidates as rc

    total_w = w_tfidf + w_skill + w_rich
    total_w = total_w if total_w > 0 else 1.0
    rc.WEIGHTS["tfidf_similarity"] = w_tfidf / total_w
    rc.WEIGHTS["skill_overlap"] = w_skill / total_w
    rc.WEIGHTS["resume_richness"] = w_rich / total_w
    return rc.WEIGHTS.copy()


def _run_screening(processed_df: pd.DataFrame, job_description: str, w_tfidf: float, w_skill: float, w_rich: float):
    _apply_weights(w_tfidf, w_skill, w_rich)
    ranked_df, jd_skills, _ = score_and_rank(processed_df, job_description, top_k=None)
    metrics = evaluate_ranking(ranked_df, jd_skills)
    metrics["band_counts"] = {
        band: int((ranked_df["final_score"].apply(_score_band) == band).sum())
        for band in ["Elite", "Strong", "Warm", "Needs review"]
    }
    metrics["top_match_count"] = int(ranked_df.iloc[0]["match_count"]) if len(ranked_df) else 0
    metrics["top_band"] = _score_band(float(ranked_df.iloc[0]["final_score"])) if len(ranked_df) else "Needs review"
    return ranked_df, jd_skills, metrics


def _candidate_brief(row: pd.Series, jd_skills: list[str]) -> str:
    matched = ", ".join(row.get("matched_skills", [])[:8]) or "none"
    missing = ", ".join(row.get("missing_skills", [])[:8]) or "none"
    return (
        f"Rank {int(row.name)} · {row.get('Category', 'Unknown')} · {float(row.get('final_score', 0)):.1f}/100\n"
        f"Matched skills: {matched}\n"
        f"Missing skills: {missing}\n"
        f"JD coverage: {int(row.get('match_count', 0))}/{len(jd_skills) if jd_skills else 0}"
    )


def _shared_skills(left_row: pd.Series, right_row: pd.Series) -> list[str]:
    left_skills = set(left_row.get("matched_skills", [])) | set(left_row.get("extra_skills", []))
    right_skills = set(right_row.get("matched_skills", [])) | set(right_row.get("extra_skills", []))
    return sorted(left_skills & right_skills)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700;800&display=swap');

:root {
    --bg: #07111F;
    --panel: rgba(15, 23, 42, 0.78);
    --panel-strong: rgba(15, 23, 42, 0.92);
    --border: rgba(148, 163, 184, 0.18);
    --text: #E2E8F0;
    --muted: #94A3B8;
    --accent: #38BDF8;
    --accent-2: #22C55E;
    --warm: #F59E0B;
    --danger: #FB7185;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 20%),
        linear-gradient(180deg, #08111F 0%, #0B1526 45%, #09111E 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.1rem;
    max-width: 1540px;
}

h1, h2, h3, h4, h5 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8, 17, 31, 0.98), rgba(15, 23, 42, 0.98));
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

div[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.panel-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem 1.1rem;
    box-shadow: 0 16px 60px rgba(15, 23, 42, 0.2);
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.75rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.34rem 0.72rem;
    border-radius: 999px;
    font-size: 0.76rem;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(15, 23, 42, 0.72);
    color: #E2E8F0;
}

.stDataFrame {
    border-radius: 18px;
    overflow: hidden;
}

.stDownloadButton button,
.stButton button {
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    font-weight: 700 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=64)
    st.markdown("### Resume Screener")
    st.caption("Command-center hiring assistant")
    st.divider()

    corpus_choice = st.selectbox(
        "Corpus source",
        ["Auto (recommended)", "Original dataset", "Fallback corpus", "Upload CSV"],
        help="Auto uses the original dataset when it exists, otherwise it falls back to the saved ranking corpus.",
    )

    uploaded_file = None
    if corpus_choice == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV with resume text", type=["csv"])

    auto_demo = st.toggle("Auto-run demo view on load", value=True, help="Shows a ranked dashboard immediately using a sample JD.")

    st.divider()
    top_k = st.slider("Leaderboard size", 5, 50, 15)
    score_threshold = st.slider("Relevant score threshold", 0, 100, 50)
    st.caption("Scoring weights")
    w_tfidf = st.slider("TF-IDF Similarity", 0.0, 1.0, 0.45, 0.05)
    w_skill = st.slider("Skill Overlap", 0.0, 1.0, 0.40, 0.05)
    w_rich = st.slider("Resume Richness", 0.0, 1.0, 0.15, 0.05)

    total_w = w_tfidf + w_skill + w_rich
    if abs(total_w - 1.0) > 0.02:
        st.warning(f"Weights sum to {total_w:.2f} — they will be normalised")

    st.divider()
    st.caption("Current corpus")
    st.code(corpus_choice, language="text")


if st.session_state.get("corpus_choice") != corpus_choice:
    for key in ["ranked", "jd_skills", "metrics", "screen_signature"]:
        st.session_state.pop(key, None)
    st.session_state["corpus_choice"] = corpus_choice


if corpus_choice == "Upload CSV":
    if uploaded_file is None:
        st.info("Upload a CSV in the sidebar to start ranking resumes.")
        processed_df = pd.DataFrame()
        source_meta = {"source": "upload", "label": "Upload CSV", "note": "Waiting for an uploaded file."}
    else:
        processed_df, source_meta = load_and_preprocess("upload", uploaded_file.read())
elif corpus_choice == "Original dataset":
    processed_df, source_meta = load_and_preprocess("builtin")
elif corpus_choice == "Fallback corpus":
    processed_df, source_meta = load_and_preprocess("fallback")
else:
    processed_df, source_meta = load_and_preprocess("auto")

if not processed_df.empty:
    _render_hero(processed_df, source_meta)

    if source_meta.get("source") != "builtin":
        st.warning(source_meta.get("note", "Using fallback data."))
    else:
        st.success(source_meta.get("note", "Loaded the original dataset."))


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("## 🎯 Executive hiring dashboard")
st.caption("Hybrid TF-IDF + skill-overlap ranking with explainability, comparison tools, and export workflows.")

tab_overview, tab_ranker, tab_compare, tab_analytics, tab_exports = st.tabs(
    ["🧭 Overview", "🏆 Rank Candidates", "🧪 Compare", "📊 Analytics", "⚙️ Exports"]
)

jd_examples = {
    "Data Scientist": (
        "We are looking for a Data Scientist with strong Python, machine learning, deep learning, TensorFlow, scikit-learn, SQL, pandas, statistics, data analysis, and data visualization skills. Experience with AWS and big data platforms preferred."
    ),
    "HR Manager": (
        "Seeking an experienced HR Manager with expertise in recruitment, talent acquisition, employee relations, payroll, HRIS, onboarding, performance management, training, and HR policy development."
    ),
    "Software Engineer": (
        "Looking for a Software Engineer with strong Java, Python, REST API, Docker, Kubernetes, AWS, CI/CD, Git, agile, and problem solving skills."
    ),
    "Finance Analyst": (
        "Finance Analyst needed with skills in financial modeling, accounting, budgeting, Excel, SQL, forecasting, risk management, ERP, SAP, and compliance."
    ),
    "Product Manager": (
        "Hiring a Product Manager with roadmap planning, stakeholder management, analytics, agile delivery, product strategy, customer discovery, and cross-functional leadership skills."
    ),
}

default_preset = st.session_state.get("preset", "Data Scientist")

with tab_ranker:
    st.subheader("Job brief and screening controls")
    preset = st.selectbox(
        "Quick-fill with an example job brief",
        list(jd_examples.keys()),
        index=list(jd_examples.keys()).index(default_preset) if default_preset in jd_examples else 0,
        key="preset",
    )
    default_jd = jd_examples.get(preset, "")
    job_description = st.text_area(
        "Paste the job description",
        value=default_jd,
        height=180,
        placeholder="Describe required skills, responsibilities, qualifications, and must-have tools.",
        key="job_description",
    )
    if not isinstance(job_description, str) or not job_description.strip() or job_description.strip().lower() == "nan":
        job_description = default_jd

    c_run, c_hint = st.columns([1, 2])
    with c_run:
        run_btn = st.button(
            "🚀 Screen and rank candidates",
            type="primary",
            use_container_width=True,
            disabled=not job_description.strip() or processed_df.empty,
        )
    with c_hint:
        st.markdown(
            "<div class='panel-card'>This dashboard now boots into a visible demo, falls back safely when the source dataset is missing, and adds ranking, comparison, and export workflows in one place.</div>",
            unsafe_allow_html=True,
        )

    should_bootstrap = auto_demo and "ranked" not in st.session_state and not processed_df.empty and bool(job_description.strip())
    if (run_btn or should_bootstrap) and not processed_df.empty and job_description.strip():
        with st.spinner("Ranking the corpus and extracting skill gaps…"):
            ranked_df, jd_skills, metrics = _run_screening(processed_df, job_description, w_tfidf, w_skill, w_rich)
            st.session_state["ranked"] = ranked_df
            st.session_state["jd_skills"] = jd_skills
            st.session_state["metrics"] = metrics
            st.session_state["screen_signature"] = (preset, job_description.strip(), corpus_choice, round(w_tfidf, 2), round(w_skill, 2), round(w_rich, 2))
            if should_bootstrap:
                st.toast("Demo ranking loaded automatically.")

    if "ranked" not in st.session_state:
        st.info("Run a ranking to unlock the leaderboard, explainability cards, and comparison view.")
    else:
        ranked_df = st.session_state["ranked"]
        jd_skills = st.session_state["jd_skills"]
        metrics = st.session_state["metrics"]

        stat_cols = st.columns(5)
        stat_cols[0].metric("Total resumes", _format_number(metrics["total_candidates"]))
        stat_cols[1].metric("Required skills", _format_number(metrics["required_skills_count"]))
        stat_cols[2].metric("Average score", f"{metrics['score_mean']:.1f}")
        stat_cols[3].metric("Top score", f"{metrics['score_max']:.1f}")
        stat_cols[4].metric("Precision@10", f"{metrics.get('precision_at_10', 0)*100:.0f}%")

        if jd_skills:
            st.markdown(
                "<div class='panel-card'><strong>Detected skills:</strong> "
                + " ".join(f"<span class='pill'>{skill}</span>" for skill in jd_skills)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No standard skills were detected in the job description, so ranking leans more heavily on semantic similarity.")

        st.markdown("### Leaderboard")
        filtered_ranked = ranked_df.copy()
        category_options = [category for category in sorted(filtered_ranked["Category"].dropna().unique().tolist())] if "Category" in filtered_ranked.columns else []
        filter_cols = st.columns([1.1, 1.1, 1.2, 1.2])
        with filter_cols[0]:
            selected_categories = st.multiselect("Category filter", category_options, default=[], help="Leave empty to include every category.")
        with filter_cols[1]:
            min_score = st.slider("Minimum score", 0, 100, score_threshold)
        with filter_cols[2]:
            skill_floor = st.slider("Minimum matched skills", 0, int(ranked_df["match_count"].max()) if len(ranked_df) else 0, 0)
        with filter_cols[3]:
            search_query = st.text_input("Search resumes or skills", placeholder="e.g. Python, HRIS, AWS")

        if selected_categories:
            filtered_ranked = filtered_ranked[filtered_ranked["Category"].isin(selected_categories)]
        filtered_ranked = filtered_ranked[filtered_ranked["final_score"] >= min_score]
        filtered_ranked = filtered_ranked[filtered_ranked["match_count"] >= skill_floor]
        if search_query.strip():
            search_mask = (
                filtered_ranked.get("resume_text", pd.Series("", index=filtered_ranked.index)).astype(str).str.contains(search_query, case=False, na=False)
                | filtered_ranked.get("skills_str", pd.Series("", index=filtered_ranked.index)).astype(str).str.contains(search_query, case=False, na=False)
            )
            filtered_ranked = filtered_ranked[search_mask]

        display_ranked = filtered_ranked.head(top_k).reset_index()
        display_cols = [c for c in ["Rank", "ID", "Category", "final_score", "match_count", "missing_count", "word_count"] if c in display_ranked.columns]
        leaderboard_df = display_ranked[display_cols].copy()
        rename_map = {
            "Rank": "Rank",
            "ID": "Candidate ID",
            "Category": "Category",
            "final_score": "Score",
            "match_count": "Skills matched",
            "missing_count": "Skills missing",
            "word_count": "Resume length",
        }
        leaderboard_df.rename(columns=rename_map, inplace=True)
        if "Score" in leaderboard_df.columns:
            leaderboard_df["Score"] = leaderboard_df["Score"].map(lambda score: f"{float(score):.1f}")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        charts_left, charts_right = st.columns(2)
        with charts_left:
            st.markdown("#### Ranking leaderboard")
            fig = plot_leaderboard(filtered_ranked, top_n=min(top_k, 15))
            if fig:
                st.pyplot(fig, use_container_width=True)
        with charts_right:
            st.markdown("#### Score distribution")
            fig = plot_score_distribution(filtered_ranked)
            if fig:
                st.pyplot(fig, use_container_width=True)

        if jd_skills:
            st.markdown("#### Skill match heatmap")
            fig = plot_skill_heatmap(filtered_ranked, jd_skills, top_n=min(top_k, 15))
            if fig:
                st.pyplot(fig, use_container_width=True)

        st.markdown("#### Candidate insight card")
        inspect_rank = st.slider("Inspect rank", 1, min(max(len(filtered_ranked), 1), top_k), 1)
        if len(filtered_ranked):
            candidate_row = filtered_ranked.reset_index().iloc[inspect_rank - 1]
            insight_cols = st.columns(3)
            with insight_cols[0]:
                st.success("Matched skills")
                st.write(", ".join(candidate_row.get("matched_skills", [])[:12]) or "—")
            with insight_cols[1]:
                st.error("Missing skills")
                st.write(", ".join(candidate_row.get("missing_skills", [])[:12]) or "None")
            with insight_cols[2]:
                st.info("Extra skills")
                st.write(", ".join(candidate_row.get("extra_skills", [])[:12]) or "—")

            st.code(_candidate_brief(candidate_row, jd_skills), language="text")

            fig_miss = plot_missing_skills(filtered_ranked, top_n_candidates=top_k)
            if fig_miss:
                st.markdown("#### Most frequently missing skills")
                st.pyplot(fig_miss, use_container_width=True)

        st.markdown("#### Export filtered leaderboard")
        export_cols = st.columns(2)
        with export_cols[0]:
            csv = filtered_ranked.reset_index().to_csv(index=False).encode()
            st.download_button(
                "Download filtered ranking CSV",
                data=csv,
                file_name="ranked_candidates_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_cols[1]:
            met_json = json.dumps(metrics, indent=2).encode()
            st.download_button(
                "Download metrics JSON",
                data=met_json,
                file_name="metrics.json",
                mime="application/json",
                use_container_width=True,
            )

with tab_overview:
    st.subheader("Command-center snapshot")
    if "metrics" in st.session_state:
        metrics = st.session_state["metrics"]
        overview_cols = st.columns(4)
        overview_cols[0].metric("Resumes scored", metrics["total_candidates"])
        overview_cols[1].metric("Ready candidates", metrics.get("band_counts", {}).get("Elite", 0) + metrics.get("band_counts", {}).get("Strong", 0))
        overview_cols[2].metric("Mean score", f"{metrics['score_mean']:.1f}")
        overview_cols[3].metric("Top fit band", metrics.get("top_band", "—"))

        band_frame = pd.DataFrame(
            {"Band": list(metrics.get("band_counts", {}).keys()), "Count": list(metrics.get("band_counts", {}).values())}
        )
        if not band_frame.empty:
            st.bar_chart(band_frame.set_index("Band"))

        top_missing = _count_missing_skills(st.session_state["ranked"], limit=10)
        if top_missing:
            st.markdown("**Priority missing skills across the current shortlist:**")
            st.markdown(
                "<div class='pill-row'>"
                + " ".join(
                    f"<span class='pill'>{skill} · {count}</span>" for skill, count in top_missing
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.get("ranked") is not None and len(st.session_state["ranked"]):
            best_row = st.session_state["ranked"].iloc[0]
            second_row = st.session_state["ranked"].iloc[1] if len(st.session_state["ranked"]) > 1 else None
            story_cols = st.columns([1.1, 1.1])
            with story_cols[0]:
                st.markdown("**Best-fit candidate**")
                st.code(_candidate_brief(best_row, st.session_state.get("jd_skills", [])), language="text")
            with story_cols[1]:
                st.markdown("**Why this works**")
                reason_lines = [
                    f"Top score: {float(best_row['final_score']):.1f}",
                    f"Matched skills: {int(best_row['match_count'])}",
                    f"Band: {_score_band(float(best_row['final_score']))}",
                ]
                if second_row is not None:
                    reason_lines.append(f"Gap to #2: {float(best_row['final_score']) - float(second_row['final_score']):.1f}")
                st.write("\n".join(f"- {line}" for line in reason_lines))
    else:
        st.info("Run a ranking to unlock the executive summary and shortlist quality card.")

with tab_compare:
    st.subheader("Candidate comparison")
    if "ranked" not in st.session_state or st.session_state["ranked"].empty:
        st.info("Run a ranking first, then compare any two shortlisted candidates.")
    else:
        ranked_df = st.session_state["ranked"]
        candidate_ranks = _candidate_options(ranked_df)
        default_candidates = candidate_ranks[:2] if len(candidate_ranks) >= 2 else candidate_ranks
        selected_candidates = st.multiselect(
            "Pick up to two ranks",
            candidate_ranks,
            default=default_candidates,
            max_selections=2,
        )
        if len(selected_candidates) < 2:
            st.info("Choose two candidate ranks to compare score components and skill overlap.")
        else:
            left_row = ranked_df.loc[selected_candidates[0]]
            right_row = ranked_df.loc[selected_candidates[1]]
            compare_cols = st.columns(2)
            for column, row in zip(compare_cols, [left_row, right_row]):
                with column:
                    st.markdown(
                        f"<div class='panel-card'><strong>Rank {int(row.name)}</strong><br>{row.get('Category', 'Unknown')}<br><span class='pill'>Score {float(row['final_score']):.1f}</span> <span class='pill'>Skills matched {int(row['match_count'])}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.write(_candidate_brief(row, st.session_state.get("jd_skills", [])))

            comparison = {
                "metric": ["final_score", "match_count", "missing_count", "word_count", "skill_overlap", "richness"],
                f"Rank {int(left_row.name)}": [
                    float(left_row.get("final_score", 0)),
                    int(left_row.get("match_count", 0)),
                    int(left_row.get("missing_count", 0)),
                    int(left_row.get("word_count", 0)),
                    float(left_row.get("skill_overlap", 0)),
                    float(left_row.get("richness", 0)),
                ],
                f"Rank {int(right_row.name)}": [
                    float(right_row.get("final_score", 0)),
                    int(right_row.get("match_count", 0)),
                    int(right_row.get("missing_count", 0)),
                    int(right_row.get("word_count", 0)),
                    float(right_row.get("skill_overlap", 0)),
                    float(right_row.get("richness", 0)),
                ],
            }
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

            shared = _shared_skills(left_row, right_row)
            shared_text = ", ".join(shared[:20]) if shared else "None"
            left_missing = ", ".join(left_row.get("missing_skills", [])[:10]) or "None"
            right_missing = ", ".join(right_row.get("missing_skills", [])[:10]) or "None"
            compare_info = st.columns(3)
            compare_info[0].success(f"Shared strengths: {shared_text}")
            compare_info[1].error(f"Rank {int(left_row.name)} missing: {left_missing}")
            compare_info[2].error(f"Rank {int(right_row.name)} missing: {right_missing}")

with tab_analytics:
    st.subheader("Data and talent analytics")
    if processed_df.empty:
        st.info("No data is loaded yet.")
    else:
        analytics_cols = st.columns(4)
        analytics_cols[0].metric("Categories", processed_df["Category"].nunique() if "Category" in processed_df.columns else "—")
        analytics_cols[1].metric("Avg word count", f"{processed_df['word_count'].mean():.0f}")
        analytics_cols[2].metric("Avg skills/resume", f"{processed_df['skill_count'].mean():.1f}")
        analytics_cols[3].metric("Top category", processed_df["Category"].value_counts().idxmax() if "Category" in processed_df.columns else "—")

        analytics_left, analytics_right = st.columns(2)
        with analytics_left:
            st.markdown("#### Category distribution")
            fig = plot_category_distribution(processed_df)
            if fig:
                st.pyplot(fig, use_container_width=True)
        with analytics_right:
            st.markdown("#### Top skills")
            fig = plot_top_skills(processed_df, top_n=20)
            if fig:
                st.pyplot(fig, use_container_width=True)

        analytics_lower_left, analytics_lower_right = st.columns(2)
        with analytics_lower_left:
            st.markdown("#### Role-wise skill demand")
            fig = plot_role_skill_demand(processed_df)
            if fig:
                st.pyplot(fig, use_container_width=True)
        with analytics_lower_right:
            st.markdown("#### Resume length distribution")
            fig = plot_resume_length(processed_df)
            if fig:
                st.pyplot(fig, use_container_width=True)

        show_cols = [c for c in ["ID", "Category", "skills_str", "skill_count", "word_count"] if c in processed_df.columns]
        st.markdown("#### Sample processed resumes")
        st.dataframe(processed_df[show_cols].head(20), use_container_width=True, hide_index=True)

with tab_exports:
    st.subheader("Exports and configuration")
    if "metrics" in st.session_state:
        st.markdown("#### Metrics JSON")
        st.json(st.session_state["metrics"])
    else:
        st.info("Run a ranking to populate metrics and export artefacts.")

    export_actions = st.columns(2)
    with export_actions[0]:
        if st.button("💾 Save all outputs to /outputs/", use_container_width=True):
            if "ranked" not in st.session_state:
                st.error("Run a screening job first.")
            else:
                save_outputs(
                    st.session_state["ranked"],
                    processed_df,
                    st.session_state["jd_skills"],
                    st.session_state["metrics"],
                    output_dir=OUTPUT_DIR,
                )
                st.success("Saved ranked_candidates.csv, extracted_skills.csv, metrics.json, and processed_resumes.csv.")
    with export_actions[1]:
        if st.session_state.get("ranked") is not None:
            ranked_export = st.session_state["ranked"].reset_index().to_csv(index=False).encode()
            st.download_button(
                "Download current ranking CSV",
                data=ranked_export,
                file_name="ranked_candidates.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("#### Scoring model")
    st.markdown(
        f"""
        - TF-IDF similarity: {w_tfidf / total_w:.0%}
        - Skill overlap: {w_skill / total_w:.0%}
        - Resume richness: {w_rich / total_w:.0%}
        - Data source: {source_meta.get('label', 'Unknown')}
        - Current corpus note: {source_meta.get('note', '')}
        """
    )
