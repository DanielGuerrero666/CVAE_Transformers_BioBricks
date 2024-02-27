import torch
import torch.nn as nn

class T1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(T1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class T2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(T2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x