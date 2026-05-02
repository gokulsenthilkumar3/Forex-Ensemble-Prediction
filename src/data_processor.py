import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, MinMaxScaler

class DataProcessor:
    def __init__(self, data_path, test_ratio=0.2, random_state=42):
        self.data_path = data_path
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler()

    def load_and_clean(self):
        """Loads data and removes outliers per currency."""
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")
        df = df.drop(columns=["currency"], errors="ignore").dropna()
        
        cleaned = []
        for code, grp in tqdm(df.groupby("currency_code"), desc="Removing Outliers"):
            q1, q3 = grp["exchange_rate"].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (grp["exchange_rate"] >= q1 - 1.5*iqr) & (grp["exchange_rate"] <= q3 + 1.5*iqr)
            cleaned.append(grp[mask])
        
        df = pd.concat(cleaned, ignore_index=True)
        return df.sort_values(["currency_code", "date"]).reset_index(drop=True)

    def split_data(self, X, y):
        """Splits scaled data into train and test sets."""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        split = int(len(X_scaled) * (1 - self.test_ratio))
        return X_scaled[:split], X_scaled[split:], y_scaled[:split], y_scaled[split:]

    @staticmethod
    def make_sequences(X, y, timesteps):
        """Creates windowed sequences for DL models."""
        Xs, ys = [], []
        for i in range(len(X) - timesteps):
            Xs.append(X[i : i + timesteps])
            ys.append(y[i + timesteps])
        return np.array(Xs), np.array(ys)
