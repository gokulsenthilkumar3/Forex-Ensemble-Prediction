import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from src.config import Config
from src.data_processor import DataProcessor
from src.feature_engine import build_feature_set
from src.model_factory import ModelFactory
from src.ensemble import StackingEnsemble
from src.utils import setup_logging, calculate_metrics, plot_results, plot_metrics_comparison

def train_pipeline():
    Config.ensure_dirs()
    log = setup_logging(Config.OUTPUT_DIR)
    log.info("Starting Forex Prediction Training Pipeline")

    # 1. Data Processing
    dp = DataProcessor(Config.DATA_PATH, Config.TEST_RATIO)
    df_raw = dp.load_and_clean()
    df_feats = build_feature_set(df_raw)
    
    feature_cols = [c for c in df_feats.columns if c not in ("date", "exchange_rate")]
    X_all = df_feats[feature_cols].values.astype(np.float32)
    y_all = df_feats["exchange_rate"].values.astype(np.float32)
    
    X_train, X_test, y_train, y_test = dp.split_data(X_all, y_all)
    log.info(f"Data Split - Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Sequence Building for DL
    X_tr_seq, y_tr_seq = dp.make_sequences(X_train, y_train, Config.TIMESTEPS)
    X_te_seq, y_te_seq = dp.make_sequences(X_test,  y_test,  Config.TIMESTEPS)
    n_feat = X_tr_seq.shape[2]

    # 3. Train DL Models
    results = []
    dl_test_preds = []
    
    dl_models = {
        "Transformer": ModelFactory.build_transformer,
        "BiLSTM_Attn": ModelFactory.build_bilstm_attn
    }

    for name, builder in dl_models.items():
        log.info(f"Training DL Model: {name}...")
        model = builder(Config.TIMESTEPS, n_feat)
        model.fit(
            X_tr_seq, y_tr_seq,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            validation_split=0.1,
            verbose=0
        )
        
        preds_sc = model.predict(X_te_seq, verbose=0).flatten()
        dl_test_preds.append(preds_sc)
        
        # Inverse transform for metrics
        y_pr = dp.scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
        y_gt = dp.scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).flatten()
        
        metrics = calculate_metrics(y_gt, y_pr, name)
        results.append(metrics)
        plot_results(y_gt, y_pr, f"{name} Predictions", os.path.join(Config.PLOTS_DIR, f"{name.lower()}.png"))
        model.save(os.path.join(Config.MODELS_DIR, f"{name.lower()}.keras"))

    # 4. Train Tree Models (Aligned with sequence length)
    n_seq = len(y_te_seq)
    X_te_flat = X_test[-n_seq:]
    y_te_flat = y_test[-n_seq:]
    
    tree_test_preds = []
    
    log.info("Training XGBoost...")
    xgb_m = xgb.XGBRegressor(**Config.XGB_PARAMS)
    xgb_m.fit(X_train, y_train.ravel())
    p_xgb = xgb_m.predict(X_te_flat)
    tree_test_preds.append(p_xgb)
    joblib.dump(xgb_m, os.path.join(Config.MODELS_DIR, "xgboost.pkl"))

    log.info("Training LightGBM...")
    lgb_m = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    lgb_m.fit(X_train, y_train.ravel())
    p_lgb = lgb_m.predict(X_te_flat)
    tree_test_preds.append(p_lgb)
    joblib.dump(lgb_m, os.path.join(Config.MODELS_DIR, "lightgbm.pkl"))

    # 5. Stacking Ensemble
    log.info("Training Stacking Ensemble...")
    ensemble = StackingEnsemble(Config.MODELS_DIR)
    stack_preds, y_stack_gt_sc = ensemble.fit(dl_test_preds, tree_test_preds, y_te_seq.flatten())
    
    y_pr_stack = dp.scaler_y.inverse_transform(stack_preds.reshape(-1, 1)).flatten()
    y_gt_stack = dp.scaler_y.inverse_transform(y_stack_gt_sc.reshape(-1, 1)).flatten()
    
    stack_metrics = calculate_metrics(y_gt_stack, y_pr_stack, "Stacking_Ensemble")
    results.append(stack_metrics)
    plot_results(y_gt_stack, y_pr_stack, "Stacking Ensemble", os.path.join(Config.PLOTS_DIR, "stacking.png"))

    # 6. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(Config.OUTPUT_DIR, "model_comparison.csv"), index=False)
    plot_metrics_comparison(results_df, os.path.join(Config.PLOTS_DIR, "metrics_heatmap.png"))
    
    log.info("Pipeline Complete!")
    log.info("\n" + results_df.to_string())

if __name__ == "__main__":
    train_pipeline()
