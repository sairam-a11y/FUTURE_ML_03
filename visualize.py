"""
visualize.py
------------
High-quality, business-friendly chart generation for the Resume Screening system.
All charts are saved as PNG and returned as Figure objects for Streamlit embedding.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

warnings.filterwarnings("ignore")

# ── Design tokens ────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#4F46E5",
    "secondary": "#7C3AED",
    "accent":    "#06B6D4",
    "success":   "#10B981",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "bg":        "#F8FAFC",
    "card":      "#FFFFFF",
    "text":      "#1E293B",
    "muted":     "#64748B",
}

BAR_COLORS = [
    "#4F46E5", "#7C3AED", "#06B6D4", "#10B981",
    "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6",
]

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors=PALETTE["muted"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=PALETTE["text"], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _save(fig, path):
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [✓] {path}")


# ── 1. Top Skills Overall ────────────────────────────────────────────────────

def plot_top_skills(processed_df: pd.DataFrame, top_n: int = 20,
                    save_path: str = None) -> plt.Figure:
    """Horizontal bar chart of most frequent skills across all resumes."""
    all_skills = [s for skills in processed_df["skills"] for s in skills]
    counts = Counter(all_skills).most_common(top_n)
    if not counts:
        return None
    labels, vals = zip(*counts)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor(PALETTE["bg"])
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(labels))]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color=PALETTE["muted"])
    _style_ax(ax, f"Top {top_n} Skills Across All Resumes",
              "Frequency", "Skill")
    if save_path:
        _save(fig, save_path)
    return fig


# ── 2. Candidate Score Comparison ───────────────────────────────────────────

def plot_score_distribution(ranked_df: pd.DataFrame,
                             save_path: str = None) -> plt.Figure:
    """Histogram of final scores with mean/median markers."""
    scores = ranked_df["final_score"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(PALETTE["bg"])
    n, bins, patches = ax.hist(scores, bins=25, color=PALETTE["primary"],
                                edgecolor="white", linewidth=0.5, alpha=0.85)
    # Colour by score range
    for patch, left in zip(patches, bins[:-1]):
        if left < 30:
            patch.set_facecolor(PALETTE["danger"])
        elif left < 55:
            patch.set_facecolor(PALETTE["warning"])
        else:
            patch.set_facecolor(PALETTE["success"])

    ax.axvline(scores.mean(),   color=PALETTE["text"],     ls="--", lw=1.5,
               label=f"Mean {scores.mean():.1f}")
    ax.axvline(scores.median(), color=PALETTE["secondary"], ls=":",  lw=1.5,
               label=f"Median {scores.median():.1f}")
    ax.legend(fontsize=9)
    _style_ax(ax, "Candidate Score Distribution",
              "Score (0–100)", "# Candidates")
    if save_path:
        _save(fig, save_path)
    return fig


# ── 3. Ranking Leaderboard ───────────────────────────────────────────────────

def plot_leaderboard(ranked_df: pd.DataFrame, top_n: int = 15,
                     save_path: str = None) -> plt.Figure:
    """Horizontal bar chart showing top-N candidates by score."""
    top = ranked_df.head(top_n).copy()
    # Use ID as label
    if "ID" in top.columns:
        labels = ["Candidate " + str(i) for i in top["ID"]]
    else:
        labels = [f"Candidate #{i}" for i in top.reset_index()["Rank"]]

    scores = top["final_score"].values
    colors = [PALETTE["success"] if s >= 60
              else PALETTE["warning"] if s >= 40
              else PALETTE["danger"] for s in scores]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor(PALETTE["bg"])
    bars = ax.barh(labels[::-1], scores[::-1],
                   color=colors[::-1], edgecolor="white")
    for bar, score in zip(bars, scores[::-1]):
        ax.text(score + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", fontsize=8.5,
                color=PALETTE["text"], fontweight="bold")
    ax.set_xlim(0, 105)
    _style_ax(ax, f"Resume Ranking Leaderboard — Top {top_n}",
              "Hybrid Score (0–100)", "Candidate")
    legend_patches = [
        mpatches.Patch(color=PALETTE["success"], label="Strong (≥60)"),
        mpatches.Patch(color=PALETTE["warning"], label="Moderate (40–60)"),
        mpatches.Patch(color=PALETTE["danger"],  label="Weak (<40)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    if save_path:
        _save(fig, save_path)
    return fig


# ── 4. Skill Match Heatmap ───────────────────────────────────────────────────

def plot_skill_heatmap(ranked_df: pd.DataFrame, jd_skills: list,
                       top_n: int = 15, save_path: str = None) -> plt.Figure:
    """Binary heatmap: top-N candidates × required skills."""
    if not jd_skills:
        return None
    top = ranked_df.head(top_n)
    skills_to_show = jd_skills[:20]

    matrix = np.zeros((len(top), len(skills_to_show)), dtype=int)
    for i, (_, row) in enumerate(top.iterrows()):
        for j, skill in enumerate(skills_to_show):
            matrix[i, j] = int(skill in (row.get("matched_skills") or []))

    candidate_labels = (
        ["C" + str(r) for r in top["ID"]]
        if "ID" in top.columns
        else [f"#{r}" for r in range(1, len(top) + 1)]
    )

    fig, ax = plt.subplots(figsize=(max(10, len(skills_to_show) * 0.7), 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(skills_to_show)))
    ax.set_xticklabels(skills_to_show, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(candidate_labels)))
    ax.set_yticklabels(candidate_labels, fontsize=8)
    ax.set_title("Skill Match Heatmap — Top Candidates vs Required Skills",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], pad=12)
    plt.colorbar(im, ax=ax, shrink=0.5, label="Has skill (1) / Missing (0)")
    if save_path:
        _save(fig, save_path)
    return fig


# ── 5. Missing Skills Chart ──────────────────────────────────────────────────

def plot_missing_skills(ranked_df: pd.DataFrame, top_n_candidates: int = 20,
                        save_path: str = None) -> plt.Figure:
    """Bar chart of most frequently missing skills among top candidates."""
    top = ranked_df.head(top_n_candidates)
    all_missing = [s for ms in top["missing_skills"] for s in ms]
    if not all_missing:
        return None
    counts = Counter(all_missing).most_common(15)
    labels, vals = zip(*counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(PALETTE["bg"])
    ax.bar(labels, vals, color=PALETTE["danger"],
           edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    _style_ax(ax, f"Most Frequently Missing Skills (Top {top_n_candidates} Candidates)",
              "Skill", "# Candidates Missing It")
    if save_path:
        _save(fig, save_path)
    return fig


# ── 6. Role-wise Skill Demand ────────────────────────────────────────────────

def plot_role_skill_demand(processed_df: pd.DataFrame, top_roles: int = 8,
                            save_path: str = None) -> plt.Figure:
    """Grouped bar: avg skill count per top role."""
    if "Category" not in processed_df.columns:
        return None
    role_stats = (
        processed_df.groupby("Category")["skill_count"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(top_roles)
    )
    roles = role_stats.index.tolist()
    means = role_stats["mean"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(PALETTE["bg"])
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(roles))]
    bars = ax.bar(roles, means, color=colors, edgecolor="white")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.1f}", ha="center", fontsize=8.5,
                color=PALETTE["text"], fontweight="bold")
    ax.set_xticklabels(roles, rotation=30, ha="right", fontsize=9)
    _style_ax(ax, "Average Skill Count by Job Role",
              "Role", "Avg. Skills per Resume")
    if save_path:
        _save(fig, save_path)
    return fig


# ── 7. Category Distribution ─────────────────────────────────────────────────

def plot_category_distribution(processed_df: pd.DataFrame,
                                save_path: str = None) -> plt.Figure:
    """Horizontal bar of resume counts by category."""
    if "Category" not in processed_df.columns:
        return None
    counts = processed_df["Category"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor(PALETTE["bg"])
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(counts))]
    ax.barh(counts.index[::-1], counts.values[::-1],
            color=colors[::-1], edgecolor="white")
    _style_ax(ax, "Resume Count by Category", "# Resumes", "Category")
    if save_path:
        _save(fig, save_path)
    return fig


# ── 8. Word-count distribution ───────────────────────────────────────────────

def plot_resume_length(processed_df: pd.DataFrame,
                       save_path: str = None) -> plt.Figure:
    """Box plot of resume word counts by top categories."""
    if "Category" not in processed_df.columns:
        return None
    top_cats = processed_df["Category"].value_counts().head(10).index
    sub = processed_df[processed_df["Category"].isin(top_cats)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor(PALETTE["bg"])
    groups = [sub[sub["Category"] == c]["word_count"].values for c in top_cats]
    bp = ax.boxplot(groups, labels=top_cats, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], BAR_COLORS * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(top_cats, rotation=35, ha="right", fontsize=8.5)
    _style_ax(ax, "Resume Word-Count Distribution by Category",
              "Category", "Word Count")
    if save_path:
        _save(fig, save_path)
    return fig


# ── Master generate-all function ─────────────────────────────────────────────

def generate_all_charts(processed_df, ranked_df, jd_skills,
                         output_dir: str = "../outputs"):
    """Generate and save every chart to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n[Generating visualizations...]")
    plot_top_skills(processed_df,      save_path=f"{output_dir}/top_skills.png")
    plot_category_distribution(processed_df, save_path=f"{output_dir}/category_dist.png")
    plot_resume_length(processed_df,   save_path=f"{output_dir}/resume_lengths.png")
    plot_score_distribution(ranked_df, save_path=f"{output_dir}/score_dist.png")
    plot_leaderboard(ranked_df,        save_path=f"{output_dir}/leaderboard.png")
    plot_skill_heatmap(ranked_df, jd_skills, save_path=f"{output_dir}/skill_heatmap.png")
    plot_missing_skills(ranked_df,     save_path=f"{output_dir}/missing_skills.png")
    plot_role_skill_demand(processed_df, save_path=f"{output_dir}/role_skill_demand.png")
    print("[✓] All charts saved.\n")
