import torch
import torch.nn as nn

class T1(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layers=6, heads=8, dropout=0.1):
        super(T1, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_dim, heads, hidden_dim, dropout), num_layers=layers)
        self.mu_layer = nn.Linear(input_dim, latent_dim)  # Capa para la media de la distribución latente
        self.logvar_layer = nn.Linear(input_dim, latent_dim)  # Capa para la desviación estándar de la distribución latente
    
    def forward(self, x):
        x = self.transformer(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class T2(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, layers=6, heads=10, dropout=0.1):
        super(T2, self).__init__()
        self.transformer = nn.TransformerDecoder(nn.TransformerDecoderLayer(latent_dim, heads, hidden_dim, dropout), num_layers=layers)
    
    def forward(self, x, memory):
        x = self.transformer(x, memory)
        return x