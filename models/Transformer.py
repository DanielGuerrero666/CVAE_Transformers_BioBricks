import torch
import torch.nn as nn

class T1(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=6, heads=8, dropout=0.1):
        super(T1, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_dim, heads, hidden_dim, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=layers)
    
    def forward(self, x):
        x = self.transformer(x)
        return x

class T2(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=6, heads=10, dropout=0.1):
        super(T2, self).__init__()
        decoder_layers = nn.TransformerDecoderLayer(input_dim, heads, hidden_dim, dropout)
        self.transformer = nn.TransformerDecoder(decoder_layers, num_layers=layers)
    
    def forward(self, tgt, memory):
        x = self.transformer(tgt, memory)
        return x