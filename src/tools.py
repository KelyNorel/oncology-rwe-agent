"""
tools.py — Survival analysis tools for the oncology RWE agent
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for Streamlit compatibility
import matplotlib.pyplot as plt
import io
import base64
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from langchain.tools import tool

DATA_PATH = "data/processed/clinical_clean.csv"

COLORS = {
    "primary":   "#2E4057",
    "secondary": "#048A81",
    "warm":      "#E76F51",
    "gray":      "#6C757D",
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["er_bin"]      = (df["er"] == "Positive").astype(int)
    df["her2_bin"]    = (df["her2"] == "Positive").astype(int)
    df["chemo_bin"]   = (df["chemo"] == "YES").astype(int)
    df["hormone_bin"] = (df["hormone_tx"] == "YES").astype(int)
    return df

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="white")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    return obj

@tool
def kaplan_meier_analysis(stratify_by: str) -> dict:
    """
    Run Kaplan-Meier survival analysis stratified by a clinical variable.
    Available variables: subtype, er, her2, hormone_tx, chemo, npi_group.
    Returns survival statistics and a base64-encoded plot.
    """
    df = load_data()

    # NPI groups
    def npi_group(score):
        if score < 2.4: return "Excellent"
        elif score < 3.4: return "Good"
        elif score < 5.4: return "Moderate"
        else: return "Poor"
    df["npi_group"] = df["npi"].apply(npi_group)

    if stratify_by not in df.columns:
        return {"error": f"Variable '{stratify_by}' not found. Available: subtype, er, her2, hormone_tx, chemo, npi_group"}

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    groups = df[stratify_by].dropna().unique()
    results = {}

    for group in groups:
        mask = df[stratify_by] == group
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "os_months"], df.loc[mask, "os_event"],
                label=f"{group} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False, lw=2)
        results[str(group)] = {
            "n": int(mask.sum()),
            "events": int(df.loc[mask, "os_event"].sum()),
            "median_os_months": float(kmf.median_survival_time_)
        }

    # Log-rank test
    valid = df.dropna(subset=[stratify_by])
    res = multivariate_logrank_test(valid["os_months"], valid[stratify_by], valid["os_event"])
    ax.text(0.05, 0.08, f"Log-rank p = {res.p_value:.4f}",
            transform=ax.transAxes, fontsize=9, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.7))

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"Kaplan-Meier — stratified by {stratify_by}", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    return make_json_serializable({
        "stratify_by": stratify_by,
        "logrank_p": float(res.p_value),
        "significant": bool(res.p_value < 0.05),
        "groups": results,
        "plot_base64": fig_to_base64(fig)
    })

@tool
def cox_model(features: list) -> dict:
    """
    Run a multivariable Cox Proportional Hazards model.
    Available features: age, grade, tumor_size, lymph_nodes, npi,
    er_bin, her2_bin, chemo_bin, hormone_bin.
    Returns hazard ratios, p-values, and C-index.
    """
    df = load_data()
    available = ["age", "grade", "tumor_size", "lymph_nodes", "npi",
                 "er_bin", "her2_bin", "chemo_bin", "hormone_bin"]

    features = [f for f in features if f in available]
    if not features:
        return {"error": f"No valid features. Available: {available}"}

    cox_df = df[["os_months", "os_event"] + features].dropna()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="os_months", event_col="os_event")

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%",
                             "exp(coef) upper 95%", "p"]].round(3)

    # Forest plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    summary_sorted = summary.sort_values("exp(coef)")
    y_pos = range(len(summary_sorted))
    colors_f = [COLORS["warm"] if p < 0.05 else COLORS["gray"]
                for p in summary_sorted["p"]]
    ax.scatter(summary_sorted["exp(coef)"], y_pos, color=colors_f, s=80, zorder=3)
    for i, (_, row) in enumerate(summary_sorted.iterrows()):
        ax.plot([row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]], [i, i],
                color=colors_f[i], lw=2)
        ax.text(row["exp(coef) upper 95%"] + 0.02, i,
                f"HR={row['exp(coef)']:.2f}", va="center", fontsize=8.5,
                color=colors_f[i])
    ax.axvline(1.0, color=COLORS["primary"], ls="--", lw=1.5, alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(summary_sorted.index)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Cox PH Model — Hazard Ratios\n(red = p<0.05)",
                 fontweight="bold", color=COLORS["primary"])
    ax.text(0.98, 0.02, f"C-index: {cph.concordance_index_:.3f}",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F4F1DE", ec=COLORS["primary"]))

    return make_json_serializable({
        "c_index": round(cph.concordance_index_, 3),
        "hazard_ratios": summary.to_dict(),
        "plot_base64": fig_to_base64(fig)
    })

@tool
def ml_prediction(cutoff_months: int = 60) -> dict:
    """
    Train ML models to predict mortality within cutoff_months.
    Default cutoff is 60 months (5 years).
    Returns AUC scores for Logistic Regression and Random Forest.
    """
    df = load_data()
    df["outcome"] = ((df["os_months"] <= cutoff_months) &
                     (df["os_event"] == 1)).astype(int)

    features = ["age", "grade", "tumor_size", "lymph_nodes",
                "er_bin", "her2_bin", "chemo_bin", "hormone_bin"]
    ml_df = df[features + ["outcome"]].dropna()
    X = ml_df[features]
    y = ml_df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200,
                                class_weight="balanced", random_state=42),
    }
    aucs = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        aucs[name] = round(roc_auc_score(y_test,
                     model.predict_proba(X_test_s)[:, 1]), 3)

    return {
        "cutoff_months": cutoff_months,
        "outcome_rate": round(y.mean(), 3),
        "n_positive": int(y.sum()),
        "auc_scores": aucs,
        "best_model": max(aucs, key=aucs.get),
        "best_auc": max(aucs.values())
    }

@tool
def describe_dataset() -> dict:
    """
    Return a summary of the METABRIC dataset including
    patient counts, event rates, and key variable distributions.
    Includes a visual overview plot.
    """
    df = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle("METABRIC Dataset Overview", fontweight="bold",
                 color=COLORS["primary"], fontsize=13)

    # 1. Overall KM
    ax = axes[0, 0]
    kmf = KaplanMeierFitter()
    kmf.fit(df["os_months"], df["os_event"], label=f"All patients (n={len(df)})")
    kmf.plot_survival_function(ax=ax, color=COLORS["primary"], ci_show=True)
    ax.set_title("Overall Survival")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")

    # 2. Subtype distribution
    ax = axes[0, 1]
    subtype_counts = df["subtype"].value_counts()
    colors_list = [COLORS["primary"], COLORS["secondary"],
                   COLORS["warm"], COLORS["gray"], "#9B5DE5", "#F72585"]
    ax.bar(subtype_counts.index, subtype_counts.values,
           color=colors_list[:len(subtype_counts)], edgecolor="white")
    ax.set_title("Molecular Subtype Distribution")
    ax.set_xlabel("Subtype")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30)

    # 3. Age distribution
    ax = axes[1, 0]
    ax.hist(df["age"].dropna(), bins=25, color=COLORS["secondary"],
            edgecolor="white", alpha=0.85)
    ax.axvline(df["age"].median(), color=COLORS["warm"],
               linestyle="--", lw=2, label=f"Median: {df['age'].median():.0f}y")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # 4. Event rate summary
    ax = axes[1, 1]
    labels = ["Deceased", "Alive/Censored"]
    values = [df["os_event"].sum(), len(df) - df["os_event"].sum()]
    ax.pie(values, labels=labels, autopct="%1.1f%%",
           colors=[COLORS["warm"], COLORS["secondary"]],
           startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("OS Event Rate")

    plt.tight_layout()

    return make_json_serializable({
        "n_patients": len(df),
        "os_event_rate": round(df["os_event"].mean(), 3),
        "median_followup_months": round(df["os_months"].median(), 1),
        "median_age": round(df["age"].median(), 1),
        "subtype_counts": df["subtype"].value_counts().to_dict(),
        "er_positive_pct": round((df["er"] == "Positive").mean(), 3),
        "her2_positive_pct": round((df["her2"] == "Positive").mean(), 3),
        "plot_base64": fig_to_base64(fig)
    })
