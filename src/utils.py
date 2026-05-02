import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def setup_logging(output_dir):
    """Sets up professional logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ForexEngine")

def calculate_metrics(y_true, y_pred, model_name):
    """Calculates a comprehensive suite of regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Directional Accuracy
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    if len(y_true_flat) > 1:
        da = np.mean(np.sign(np.diff(y_true_flat)) == np.sign(np.diff(y_pred_flat))) * 100
    else:
        da = 0.0
        
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        "Model": model_name,
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "DA": float(da)
    }

def plot_results(y_true, y_pred, title, save_path):
    """Generates a high-quality actual vs predicted plot."""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:200], label="Actual", alpha=0.8, lw=1.5)
    plt.plot(y_pred[:200], label="Predicted", alpha=0.8, lw=1.2, linestyle="--")
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_metrics_comparison(results_df, save_path):
    """Plots a heatmap comparing all model performances."""
    plt.figure(figsize=(10, 6))
    metrics_pivot = results_df.set_index("Model")[["MAE", "RMSE", "R2", "DA"]]
    # Normalize for color scaling while keeping annotations original
    norm_metrics = (metrics_pivot - metrics_pivot.min()) / (metrics_pivot.max() - metrics_pivot.min() + 1e-10)
    
    sns.heatmap(norm_metrics, annot=metrics_pivot, fmt=".4f", cmap="RdYlGn_r", linewidths=0.5)
    plt.title("Model Metrics Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
