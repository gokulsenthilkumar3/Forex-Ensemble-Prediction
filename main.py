"""
main.py
=======
Entry point for the Forex Ensemble Prediction pipeline.

Usage
-----
    python main.py
    python main.py --data path/to/Forex_Data.csv --output results/ --timesteps 20
    DATA_PATH=Forex_Data.csv OUTPUT_DIR=results python main.py

All parameters can also be set via environment variables.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import joblib
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge
from tqdm import tqdm

# ── Project modules ───────────────────────────────────────────────────────────
from src.data.loader         import load_forex_data
from src.data.cleaner        import remove_outliers_iqr
from src.features.engineer   import add_features, load_config
from src.features.cross_currency import add_cross_currency_features
from src.features.shap_ranking   import compute_shap_ranking
from src.models.tree_models  import build_xgboost, build_lightgbm, train_xgboost, train_lightgbm, save_model
from src.models.deep_learning import DL_BUILDERS, get_callbacks
from src.evaluation.metrics  import compute_metrics
from src.evaluation.visualize import plot_loss_curves, plot_metrics_heatmap, plot_actual_vs_predicted
from src.utils.io            import make_sequences, save_results, save_scaler


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forex Ensemble Prediction Pipeline v3.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       default=os.environ.get("DATA_PATH",  "Forex_Data.csv"),
                   help="Path to input CSV dataset.")
    p.add_argument("--output",     default=os.environ.get("OUTPUT_DIR", "outputs"),
                   help="Directory for all output artifacts.")
    p.add_argument("--config",     default=os.environ.get("FEAT_CONFIG", "config/features.yaml"),
                   help="Path to features YAML config.")
    p.add_argument("--timesteps",  type=int, default=15,
                   help="Lookback window for sequence models.")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="Fraction of data held out for testing.")
    p.add_argument("--epochs",     type=int, default=40,
                   help="Max training epochs for DL models.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Batch size for DL model training.")
    p.add_argument("--patience",   type=int, default=7,
                   help="Early stopping patience (epochs).")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity level.")
    return p.parse_args()


def setup_logging(level: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log"), mode="w"),
        ],
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level, args.output)
    log = logging.getLogger(__name__)
    sns.set_theme(style="darkgrid", palette="muted")

    log.info("=" * 70)
    log.info("Forex Ensemble Prediction Pipeline v3.1")
    log.info(f"  data       : {args.data}")
    log.info(f"  output     : {args.output}")
    log.info(f"  config     : {args.config}")
    log.info(f"  timesteps  : {args.timesteps}")
    log.info(f"  test_ratio : {args.test_ratio}")
    log.info(f"  epochs     : {args.epochs}")
    log.info(f"  seed       : {args.seed}")
    log.info("=" * 70)

    feat_cfg = load_config(args.config)

    # ── 1. Load & Clean ────────────────────────────────────────────────────
    df = load_forex_data(args.data)
    df = remove_outliers_iqr(df)

    # ── 2. Feature Engineering ──────────────────────────────────────────────
    log.info("STEP 2: Feature Engineering")
    dfs = []
    for code, grp in tqdm(df.groupby("currency_code"), desc="Features"):
        g = add_features(grp, feat_cfg)
        g["currency_code"] = code
        dfs.append(g)
    df = pd.concat(dfs, ignore_index=True).sort_values(["currency_code", "date"]).reset_index(drop=True)

    # ── 3. Cross-Currency Features ───────────────────────────────────────────
    df = add_cross_currency_features(df, feat_cfg["cross_currency"])
    df = df.dropna().reset_index(drop=True)

    # ── 4. Target Engineering ───────────────────────────────────────────────
    target_mode = feat_cfg["target"]["mode"]
    n_ahead     = feat_cfg["target"]["n_steps_ahead"]
    log.info(f"STEP 4: Target Engineering (mode={target_mode})")
    if target_mode == "raw_rate":
        df["target"] = df["exchange_rate"]
    elif target_mode == "log_return":
        df["target"] = df.groupby("currency_code")["exchange_rate"].transform(
            lambda p: np.log(p / p.shift(1)))
    elif target_mode == "pct_change":
        df["target"] = df.groupby("currency_code")["exchange_rate"].transform(
            lambda p: p.pct_change())
    elif target_mode == "n_step_ahead":
        df["target"] = df.groupby("currency_code")["exchange_rate"].transform(
            lambda p: p.shift(-n_ahead))
    else:
        raise ValueError(f"Unknown target mode: {target_mode}")
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # ── 5. Per-Currency Scaling ─────────────────────────────────────────────
    log.info("STEP 5: Per-Currency Scaling")
    feature_cols = [c for c in df.columns if c not in ("date", "exchange_rate", "target", "currency_code")]
    per_currency_scalers = {}
    scaled_dfs = []
    for code, grp in tqdm(df.groupby("currency_code"), desc="Scaling"):
        sc = RobustScaler()
        cols = [c for c in feature_cols if c in grp.columns]
        grp = grp.copy()
        grp[cols] = sc.fit_transform(grp[cols].values)
        per_currency_scalers[code] = {"scaler": sc, "cols": cols}
        scaled_dfs.append(grp)
    df = pd.concat(scaled_dfs, ignore_index=True).sort_values(["currency_code", "date"]).reset_index(drop=True)
    save_scaler(per_currency_scalers, args.output, "per_currency_scalers")

    df = pd.get_dummies(df, columns=["currency_code"], drop_first=True)
    bool_cols = df.select_dtypes("bool").columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    # ── 6. Target Scaling ─────────────────────────────────────────────────────
    log.info("STEP 6: Target Scaling & Train/Test Split")
    feature_cols_final = [c for c in df.columns if c not in ("date", "exchange_rate", "target")]
    X_all = df[feature_cols_final].values.astype(np.float32)
    y_all = df["target"].values.reshape(-1, 1).astype(np.float32)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_all)
    save_scaler(scaler_y, args.output, "scaler_y")

    split       = int(len(X_all) * (1 - args.test_ratio))
    X_train, X_test = X_all[:split],    X_all[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    X_tr_flat, y_tr_flat = X_train, y_train.ravel()
    X_te_flat, y_te_flat = X_test,  y_test.ravel()
    log.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    X_tr_seq, y_tr_seq = make_sequences(X_train, y_train, args.timesteps)
    X_te_seq, y_te_seq = make_sequences(X_test,  y_test,  args.timesteps)
    n_feat = X_tr_seq.shape[2]
    log.info(f"Seq train: {X_tr_seq.shape}  |  Seq test: {X_te_seq.shape}")

    def inv(arr):
        return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

    # ── 7. Tree Models ──────────────────────────────────────────────────────
    log.info("STEP 7: Training Tree Models")
    xgb_model = train_xgboost(build_xgboost(), X_tr_flat, y_tr_flat, X_te_flat, y_te_flat)
    lgb_model = train_lightgbm(build_lightgbm(), X_tr_flat, y_tr_flat, X_te_flat, y_te_flat)
    save_model(xgb_model, os.path.join(args.output, "xgb_model.pkl"))
    save_model(lgb_model, os.path.join(args.output, "lgb_model.pkl"))

    # ── 8. SHAP Ranking ────────────────────────────────────────────────────────
    log.info("STEP 8: SHAP Feature Ranking")
    compute_shap_ranking(
        model=lgb_model, X=X_te_flat[:500],
        feature_names=feature_cols_final,
        output_dir=args.output, top_n=30,
    )

    # ── 9. Deep Learning Models ───────────────────────────────────────────────
    log.info("STEP 9: Training Deep Learning Models")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        [tf.config.experimental.set_memory_growth(d, True) for d in gpus]

    results, histories, preds, dl_preds = [], {}, {}, {}
    for name, builder_fn in DL_BUILDERS.items():
        log.info(f"  Training {name}...")
        model = builder_fn(args.timesteps, n_feat)
        hist  = model.fit(
            X_tr_seq, y_tr_seq,
            epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.2,
            callbacks=get_callbacks(name, args.output, args.patience),
            verbose=0,
        )
        p_sc = model.predict(X_te_seq, batch_size=args.batch_size, verbose=0).flatten()
        dl_preds[name] = p_sc
        y_pr = inv(p_sc)
        y_tr = inv(y_te_seq.flatten())
        m    = compute_metrics(y_tr, y_pr, name)
        results.append(m)
        histories[name] = hist
        preds[name]     = (y_pr, y_tr)
        model.save(os.path.join(args.output, f"{name.lower().replace('-','_')}.keras"))
        log.info(f"    {name}: MAE={m['MAE']:.4f}  R²={m['R2']:.4f}  DA={m['DA']:.1f}%")

    # ── 10. Tree Evaluation ────────────────────────────────────────────────────
    log.info("STEP 10: Tree Model Evaluation")
    n_seq     = len(y_te_seq)
    X_te_algn = X_te_flat[-n_seq:]
    y_te_algn = y_te_flat[-n_seq:]
    for tname, tmodel in [("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
        p_sc   = tmodel.predict(X_te_algn).astype(np.float32)
        y_pr   = inv(p_sc)
        y_tr   = inv(y_te_algn)
        m      = compute_metrics(y_tr, y_pr, tname)
        results.append(m)
        preds[tname] = (y_pr, y_tr)
        log.info(f"  {tname}: MAE={m['MAE']:.4f}  R²={m['R2']:.4f}  DA={m['DA']:.1f}%")

    # ── 11. Stacking Ensemble ──────────────────────────────────────────────────
    log.info("STEP 11: Stacking Ensemble")
    meta_X = np.column_stack([
        *[dl_preds[n] for n in DL_BUILDERS],
        xgb_model.predict(X_te_algn),
        lgb_model.predict(X_te_algn),
    ])
    meta_y = y_te_seq.flatten()
    mid    = int(len(meta_X) * 0.5)
    meta_lr = Ridge(alpha=1.0)
    meta_lr.fit(meta_X[:mid], meta_y[:mid])
    save_model(meta_lr, os.path.join(args.output, "stacking_meta.pkl"))
    st_pred  = meta_lr.predict(meta_X[mid:])
    m_stack  = compute_metrics(inv(meta_y[mid:]), inv(st_pred), "Stacking Ensemble")
    results.append(m_stack)
    preds["Stacking Ensemble"] = (inv(st_pred), inv(meta_y[mid:]))
    log.info(f"  Stacking: MAE={m_stack['MAE']:.4f}  R²={m_stack['R2']:.4f}  DA={m_stack['DA']:.1f}%")

    # ── 12. Results & Visualizations ─────────────────────────────────────────────
    log.info("STEP 12: Results & Visualizations")
    results_df = pd.DataFrame(results)
    save_results(results, args.output)
    log.info("\n" + results_df.to_string(index=False))
    plot_loss_curves(histories, args.output)
    plot_metrics_heatmap(results_df, args.output)
    plot_actual_vs_predicted(preds, args.output)

    best = min(results, key=lambda r: r["MAE"])
    log.info("=" * 70)
    log.info(f"BEST MODEL: {best['Model']}")
    for k in ["MSE", "MAE", "RMSE", "MAPE", "R2", "DA"]:
        log.info(f"  {k:5s}: {best[k]:.6f}")
    log.info("=" * 70)
    log.info(f"Outputs → {os.path.abspath(args.output)}")
    log.info("Forex Prediction v3.1 complete ✓")


if __name__ == "__main__":
    main()
