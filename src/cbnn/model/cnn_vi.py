
from typing import List, Optional

from .base_vae import BaseVAE, BaseVariationalBiasedAutoEncoder
from .modules.encoders import CNNVariationalEncoder
from .modules.decoders import CNNVariationalDecoder
from .modules.classifiers import MLPClassifier
from .modules.classifiers import BayesianClassifier
from ..trainer.losses import normal_kullback_leibler_divergence

import torch



class CNNVAE(BaseVAE):
    """
    Modified from https://github.com/AntixK/PyTorch-VAE
    """

    def _init_modules(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        
        self.latent_dim = latent_dim

        # Build Encoder
        self.encoder = CNNVariationalEncoder(in_channels, latent_dim, hidden_dims)

        # Build Decoder
        if hidden_dims is not None:
            hidden_dims.reverse()
            
        self.decoder = CNNVariationalDecoder(latent_dim, in_channels, hidden_dims)




class CNNVariationalBiasedAutoEncoder(BaseVariationalBiasedAutoEncoder):

    def _init_modules(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:

        self.latent_dim = latent_dim
        self.encoder = CNNVariationalEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = CNNVariationalDecoder(latent_dim, in_channels, 2 * hidden_dims)
    



class CNNVariationalBiasedInferer(BaseVAE):
    """
    Variational Context-Biased Inference model
    """

    def _init_modules(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 encoder_hidden_dims: List = None,
                 classifier_hidden_dim: int = 128,
                 classifier_nb_layers: int = 3,
                 **kwargs) -> None:

        self.latent_dim = latent_dim
        self.encoder = CNNVariationalEncoder(in_channels, latent_dim, encoder_hidden_dims)
        self.decoder = MLPClassifier(latent_dim, out_channels, classifier_hidden_dim, classifier_nb_layers)
    
    
    def loss_function(self, y_target : torch.Tensor, y_recon : torch.Tensor, mu : torch.Tensor, log_var : torch.Tensor, **kwargs):
        kld_weight = kwargs['M_N']

        infer_loss = torch.nn.functional.cross_entropy(y_recon, y_target, reduction='sum') # use CrossEntropy as classification loss instead or MSE reconstruction loss
        kld_loss = normal_kullback_leibler_divergence(mu, log_var)

        loss = infer_loss + kld_weight * kld_loss
        return {'loss': loss, 'Inference_Loss':infer_loss.detach(), 'KLD':-kld_loss.detach()}
        

