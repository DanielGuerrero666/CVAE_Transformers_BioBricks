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
        # Obtener el output del encoder para usarlo como memory en el decoder
        memory = z.unsqueeze(0).expand(self.decoder.transformer.num_layers, -1, -1)
        recon_x = self.decoder(x, memory)
        return recon_x, z.mean(), z.var()