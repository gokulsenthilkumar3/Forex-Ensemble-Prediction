import joblib
import os
import numpy as np
from sklearn.linear_model import Ridge

class StackingEnsemble:
    def __init__(self, output_dir, alpha=1.0):
        self.output_dir = output_dir
        self.meta_learner = Ridge(alpha=alpha)
        self.save_path = os.path.join(output_dir, "stacking_meta.pkl")

    def fit(self, dl_preds, tree_preds, y_true):
        """
        Fits the meta-learner on predictions from base models.
        
        Args:
            dl_preds: List of arrays containing DL model predictions.
            tree_preds: List of arrays containing Tree model predictions.
            y_true: Ground truth values.
        """
        # Align lengths and stack meta-features
        X_meta = np.column_stack(dl_preds + tree_preds)
        
        # Simple split for meta-learner training to avoid leakage
        split = int(len(X_meta) * 0.7)
        self.meta_learner.fit(X_meta[:split], y_true[:split])
        
        joblib.dump(self.meta_learner, self.save_path)
        return self.predict(X_meta[split:]), y_true[split:]

    def predict(self, X_meta):
        """Generates final prediction using the trained meta-learner."""
        return self.meta_learner.predict(X_meta)

    def load(self):
        """Loads a pre-trained meta-learner."""
        if os.path.exists(self.save_path):
            self.meta_learner = joblib.load(self.save_path)
        else:
            raise FileNotFoundError(f"No meta-learner found at {self.save_path}")
