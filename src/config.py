import os

class Config:
    # Paths
    DATA_PATH = "Forex_Data.csv"
    OUTPUT_DIR = "outputs"
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    
    # Model Hyperparameters
    TIMESTEPS = 15
    TEST_RATIO = 0.2
    RANDOM_STATE = 42
    
    # Deep Learning Settings
    BATCH_SIZE = 256
    EPOCHS = 40
    PATIENCE = 7
    
    # Tree Model Settings
    XGB_PARAMS = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": -1
    }
    
    LGB_PARAMS = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "verbosity": -1
    }

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.MODELS_DIR, cls.PLOTS_DIR]:
            os.makedirs(d, exist_ok=True)
