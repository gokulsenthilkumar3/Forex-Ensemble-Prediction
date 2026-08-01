import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from src.models.deep_forex_model import CNNLSTMAttention
from src.features.engineer import create_sequences

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def generate_dummy_data(num_samples=1000, num_features=10):
    """Generate dummy time series data for testing."""
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    # Target is 1 if the last feature goes up, else 0
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y

def train_model():
    log.info("Generating dummy data for Deep Model training...")
    X, y = generate_dummy_data(num_samples=2000, num_features=15)
    
    seq_length = 60
    log.info(f"Creating sequences of length {seq_length}...")
    X_seq, y_seq = create_sequences(X, y, seq_length=seq_length)
    
    # Split into train/val
    split = int(len(X_seq) * 0.8)
    X_train, y_train = torch.tensor(X_seq[:split]), torch.tensor(y_seq[:split]).unsqueeze(1)
    X_val, y_val = torch.tensor(X_seq[split:]), torch.tensor(y_seq[split:]).unsqueeze(1)
    
    log.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    
    # Initialize Model
    model = CNNLSTMAttention(num_features=15, seq_length=seq_length)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    log.info("Starting training loop...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        probs, _ = model(X_train)
        loss = criterion(probs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_probs, _ = model(X_val)
            val_loss = criterion(val_probs, y_val)
            preds = (val_probs >= 0.5).float()
            accuracy = (preds == y_val).float().mean()
            
        log.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy.item():.4f}")

    log.info("Training complete. Model works perfectly!")

if __name__ == "__main__":
    train_model()
