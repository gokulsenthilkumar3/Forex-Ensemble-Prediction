"""
src/evaluation/visualize.py
============================
Visualization helpers: loss curves, metrics heatmap, actual vs predicted.
"""

from __future__ import annotations
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)


def plot_loss_curves(
    histories: dict,
    output_dir: str,
    filename: str = "loss_curves.png",
) -> None:
    """
    Plot training vs validation Huber loss for each DL model.

    Parameters
    ----------
    histories   : Dict mapping model name -> Keras History object.
    output_dir  : Directory to save the PNG.
    filename    : Output filename.
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist.history["loss"],     label="Train")
        ax.plot(hist.history["val_loss"], label="Val", linestyle="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Huber Loss")
    plt.suptitle("Training & Validation Loss", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Loss curves saved to {path}")


def plot_metrics_heatmap(
    results_df: pd.DataFrame,
    output_dir: str,
    filename: str = "metrics_heatmap.png",
) -> None:
    """
    Render a color-normalized heatmap of all model metrics.

    Parameters
    ----------
    results_df : DataFrame with columns [Model, MAE, RMSE, MAPE, R2, DA].
    output_dir : Directory to save the PNG.
    filename   : Output filename.
    """
    pivot = results_df.set_index("Model")[["MAE", "RMSE", "MAPE", "R2", "DA"]].astype(float)
    norm  = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-10)
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) // 2)))
    sns.heatmap(
        norm, annot=pivot.round(4), fmt="",
        cmap="RdYlGn", ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Normalized Score"},
    )
    ax.set_title("Model Metrics Heatmap (values annotated)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Metrics heatmap saved to {path}")


def plot_actual_vs_predicted(
    preds: dict,
    output_dir: str,
    n_samples: int = 300,
) -> None:
    """
    Save an actual-vs-predicted line chart for every model.

    Parameters
    ----------
    preds     : Dict mapping model name -> (y_pred, y_true) tuple of 1-D arrays.
    output_dir: Directory to save PNGs.
    n_samples : Max number of samples to plot per chart.
    """
    for name, (y_pred, y_true) in preds.items():
        n = min(n_samples, len(y_true))
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(y_true[:n], label="Actual",    lw=1.5)
        ax.plot(y_pred[:n], label="Predicted", lw=1.0, alpha=0.85, linestyle="--")
        ax.set_title(f"{name} – Actual vs Predicted", fontsize=11)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        plt.tight_layout()
        fname = f"pred_{name.lower().replace(' ', '_').replace('-', '_')}.png"
        path  = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
    log.info(f"Actual vs predicted charts saved to {output_dir}")
