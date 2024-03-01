import torch
import torch.nn as nn

class T1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers = 6, heads = 8, dropout = 0.1):
        super(T1, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_dim, heads, hidden_dim, dropout), layers)
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.transformer(x)
        x = torch.relu(self.fc1(x))
        return x

class T2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers = 6, heads = 10, dropout = 0.1):
        super(T2, self).__init__()
        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(input_dim, heads, hidden_dim, dropout), layers)
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, tgt, memory):
        x = self.transformer(tgt, memory)
        x = torch.relu(self.fc1(x))
        return x