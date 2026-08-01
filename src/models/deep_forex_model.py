import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # (batch_size, seq_len, 1)
        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        return context, attn_weights

class CNNLSTMAttention(nn.Module):
    def __init__(self, num_features, seq_length, cnn_filters=64, lstm_hidden=128, dropout_rate=0.3):
        super(CNNLSTMAttention, self).__init__()
        
        self.num_features = num_features
        self.seq_length = seq_length
        
        # 1D CNN for spatial feature extraction across time
        # Conv1d expects (batch_size, in_channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # LSTM for temporal sequence processing
        # LSTM expects (batch_size, seq_len, input_size) if batch_first=True
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                            num_layers=2, batch_first=True, dropout=dropout_rate if dropout_rate > 0 else 0)
        
        # Temporal Attention Mechanism
        self.attention = Attention(lstm_hidden)
        
        # Fully Connected Layers for Classification (e.g. Buy/Sell/Hold or probability)
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Binary classification (Buy=1, Sell=0) or multi-class. We use 1 output for probability.
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        
        # Reshape for Conv1d: (batch_size, num_features, seq_len)
        x = x.transpose(1, 2)
        
        # CNN layer
        c = self.conv1(x)
        c = self.relu(c)
        c = self.dropout1(c)
        
        # Reshape for LSTM: (batch_size, seq_len, cnn_filters)
        c = c.transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(c) # lstm_out: (batch_size, seq_len, lstm_hidden)
        
        # Attention layer
        context, attn_weights = self.attention(lstm_out)
        
        # Fully Connected
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Output probability
        out = self.fc2(out)
        prob = self.sigmoid(out)
        
        return prob, attn_weights

    def predict(self, x, threshold=0.5):
        """
        Inference helper to return binary signals based on probability threshold.
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            prob, _ = self.forward(x)
            predictions = (prob >= threshold).float()
            return predictions.cpu().numpy(), prob.cpu().numpy()
