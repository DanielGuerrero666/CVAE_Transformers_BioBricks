import torch
import torch.nn as nn
from models.Transformer import T1, T2

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(CVAE, self).__init__()
        self.encoder = T1(input_dim, hidden_dim, latent_dim)
        self.decoder = T2(latent_dim, hidden_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        recon_x_sigmoid = torch.sigmoid(recon_x)
        return recon_x_sigmoid