
from typing import Optional, List

import torch


class CNNVariationalEncoder(torch.nn.Module):
    """
    Modified from the vanilla VAE at https://github.com/AntixK/PyTorch-VAE
    """

    def __init__(self,
                 in_channels: int,
                 image_dim: int,
                 latent_dim: int,
                 hidden_dims: Optional[List] = None,
                 **kwargs) -> None:
        super(CNNVariationalEncoder, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        self.reduced_dims = image_dim // 2**len(hidden_dims)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    torch.nn.BatchNorm2d(h_dim),
                    torch.nn.SiLU())
            )
            in_channels = h_dim

        self.encoder = torch.nn.Sequential(*modules)
        self.fc_mu = torch.nn.Linear(hidden_dims[-1]*self.reduced_dims**2, latent_dim)
        self.fc_var = torch.nn.Linear(hidden_dims[-1]*self.reduced_dims**2, latent_dim)        


    def forward(self, x: torch.Tensor):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]