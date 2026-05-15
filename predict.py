"""
predict.py
==========
Inference script for the Forex Ensemble Prediction pipeline.

Loads scalers, models, and produces predictions for new input data.

Usage
-----
    python predict.py --input new_data.csv --output predictions.csv
    python predict.py --input new_data.csv --model-dir outputs/latest/
    python predict.py --input new_data.csv --model-dir outputs/run_20260515_221500/
"""

from __future__ import annotations
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import joblib

from src.data.loader       import load_forex_data
from src.data.cleaner      import remove_outliers_iqr
from src.features.engineer import add_features, load_config
from src.features.cross_currency import add_cross_currency_features

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forex Ensemble Prediction — Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     required=True,  help="Path to new CSV data for inference.")
    p.add_argument("--output",    default="predictions.csv", help="Output CSV path.")
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "outputs/latest"),
                   help="Directory containing saved models and scalers.")
    p.add_argument("--config",    default="config/features.yaml", help="Feature config YAML path.")
    p.add_argument("--model",     default="lgb",
                   choices=["lgb", "xgb", "stacking"],
                   help="Which model to use for prediction.")
    return p.parse_args()


def load_artifacts(model_dir: str, model_name: str) -> tuple:
    """
    Load scaler_y, per_currency_scalers, and the selected model from model_dir.

    Parameters
    ----------
    model_dir  : Directory containing saved .pkl artifacts.
    model_name : One of 'lgb', 'xgb', 'stacking'.

    Returns
    -------
    (scaler_y, per_currency_scalers, model)
    """
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    per_currency_scalers = joblib.load(os.path.join(model_dir, "per_currency_scalers.pkl"))

    model_file_map = {
        "lgb":      "lgb_model.pkl",
        "xgb":      "xgb_model.pkl",
        "stacking": "stacking_meta.pkl",
    }
    model_path = os.path.join(model_dir, model_file_map[model_name])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    log.info(f"Loaded model: {model_path}")
    return scaler_y, per_currency_scalers, model


def preprocess(df: pd.DataFrame, feat_cfg: dict, per_currency_scalers: dict) -> tuple:
    """
    Apply the same feature engineering + per-currency scaling pipeline
    used during training to new inference data.

    Parameters
    ----------
    df                   : Raw loaded DataFrame.
    feat_cfg             : Loaded features.yaml config.
    per_currency_scalers : Dict of {currency_code: {scaler, cols}} from training.

    Returns
    -------
    (X, feature_cols, df_processed)
    """
    df = remove_outliers_iqr(df)

    dfs = []
    for code, grp in df.groupby("currency_code"):
        g = add_features(grp, feat_cfg)
        g["currency_code"] = code
        dfs.append(g)
    df = pd.concat(dfs, ignore_index=True).sort_values(["currency_code", "date"]).reset_index(drop=True)
    df = add_cross_currency_features(df, feat_cfg["cross_currency"])
    df = df.dropna().reset_index(drop=True)

    scaled_dfs = []
    for code, grp in df.groupby("currency_code"):
        grp = grp.copy()
        if code in per_currency_scalers:
            sc   = per_currency_scalers[code]["scaler"]
            cols = [c for c in per_currency_scalers[code]["cols"] if c in grp.columns]
            grp[cols] = sc.transform(grp[cols].values)
        else:
            log.warning(f"No scaler found for currency {code}. Using unscaled features.")
        scaled_dfs.append(grp)
    df = pd.concat(scaled_dfs, ignore_index=True)

    df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
    bool_cols = df.select_dtypes("bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    feature_cols = [c for c in df.columns if c not in ("date", "exchange_rate", "target")]
    X = df[feature_cols].values.astype(np.float32)
    return X, feature_cols, df


def main() -> None:
    args = parse_args()
    feat_cfg = load_config(args.config)

    log.info(f"Loading input data: {args.input}")
    df_raw = load_forex_data(args.input)

    log.info(f"Loading artifacts from: {args.model_dir}")
    scaler_y, per_currency_scalers, model = load_artifacts(args.model_dir, args.model)

    X, feature_cols, df_proc = preprocess(df_raw, feat_cfg, per_currency_scalers)
    log.info(f"Inference input shape: {X.shape}")

    preds_scaled = model.predict(X).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

    out_df = df_proc[["date", "exchange_rate"]].copy() if "exchange_rate" in df_proc.columns else df_proc[["date"]].copy()
    out_df["predicted_rate"] = preds

    out_df.to_csv(args.output, index=False)
    log.info(f"Predictions saved to {args.output} ({len(out_df):,} rows).")


if __name__ == "__main__":
    main()
