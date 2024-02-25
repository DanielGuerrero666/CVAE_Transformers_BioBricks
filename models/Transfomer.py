import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_layers = TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers)
        
    def forward(self, src):
        # src: tensor de forma (seq_length, batch_size, input_dim)
        output = self.transformer_encoder(src)
        return output
