"""
src/features/shap_ranking.py
=============================
SHAP-based feature importance ranking.
Writes a sorted CSV and optional bar chart to the output directory.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

log = logging.getLogger(__name__)


def compute_shap_ranking(
    model,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    top_n: int = 30,
    save_plot: bool = True,
) -> pd.DataFrame:
    """
    Compute SHAP-based feature importance for a tree model and
    write results to CSV + optional bar chart PNG.

    Parameters
    ----------
    model        : Fitted LightGBM / XGBoost model (TreeExplainer compatible)
    X            : Feature array used for SHAP computation (use test/val set)
    feature_names: List of feature column names aligned with X columns
    output_dir   : Directory to save outputs
    top_n        : Number of top features to show in chart
    save_plot    : Whether to save a bar chart PNG

    Returns
    -------
    DataFrame with columns [feature, mean_shap, rank] sorted descending by importance
    """
    try:
        import shap
    except ImportError:
        log.error("shap not installed. Run: pip install shap")
        return pd.DataFrame()

    log.info(f"Computing SHAP values on {len(X)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_shap = np.abs(shap_values).mean(axis=0)
    ranking_df = pd.DataFrame({
        "feature":   feature_names,
        "mean_shap": mean_shap,
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    csv_path = os.path.join(output_dir, "features_ranked.csv")
    ranking_df.to_csv(csv_path, index=False)
    log.info(f"Feature ranking saved to {csv_path}")

    # Low-importance features (bottom 20% by SHAP)
    threshold = ranking_df["mean_shap"].quantile(0.2)
    low_importance = ranking_df[ranking_df["mean_shap"] <= threshold]["feature"].tolist()
    log.info(f"Low-importance features (bottom 20%): {low_importance}")

    low_imp_path = os.path.join(output_dir, "features_low_importance.txt")
    with open(low_imp_path, "w") as f:
        f.write("\n".join(low_importance))
    log.info(f"Low-importance feature list saved to {low_imp_path}")

    if save_plot:
        top = ranking_df.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n // 2)))
        sns.barplot(
            x="mean_shap", y="feature", data=top, ax=ax, palette="viridis"
        )
        ax.set_title(f"Top {top_n} Features by SHAP Importance", fontsize=13)
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "shap_importance.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log.info(f"SHAP chart saved to {plot_path}")

    return ranking_df
