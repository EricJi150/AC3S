import torch.nn as nn

class Modulator(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.2, min_scale=0.3, max_scale=1.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.fc1 = nn.Linear(26, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc3(y)
        y = self.sigmoid(y)
        y = self.min_scale + (self.max_scale - self.min_scale) * y
        return y