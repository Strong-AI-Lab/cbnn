
from typing import List

import torch


class CNNVariationalDecoder(torch.nn.Module):
    """
    Modified from the vanilla VAE at https://github.com/AntixK/PyTorch-VAE
    """

    def __init__(self,
                 latent_dim: int,
                 in_channels: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(CNNVariationalDecoder, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        # Build Decoder
        modules = []

        self.decoder_input = torch.nn.Linear(latent_dim, hidden_dims * 4)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    torch.nn.BatchNorm2d(hidden_dims[i + 1]),
                    torch.nn.LeakyReLU())
            )

        self.decoder = torch.nn.Sequential(*modules)

        self.final_layer = torch.nn.Sequential(
                            torch.nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            torch.nn.BatchNorm2d(hidden_dims[-1]),
                            torch.nn.LeakyReLU(),
                            torch.nn.Conv2d(hidden_dims[-1], out_channels=in_channels,
                                      kernel_size= 3, padding= 1),
                            torch.nn.Tanh())
    

    def forward(self, z: torch.Tensor):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
