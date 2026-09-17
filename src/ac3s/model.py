import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(27, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        return y

model = MLP().to("cuda") 
model.train()