import torch
import torch.nn as nn
from models.Transformer import T1, T2

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(CVAE, self).__init__()
        self.encoder = T1(input_dim, hidden_dim, latent_dim)
        self.decoder = T2(latent_dim, hidden_dim, output_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)
        recon_x = self.decoder(z,z)
        return recon_x, mu, logvar

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std