
from typing import Optional

from ..trainer.losses import normal_kullback_leibler_divergence

import torch
import pytorch_lightning as pl


class BaseVAE(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super(BaseVAE, self).__init__()
        self.encoder = None
        self.decoder = None

        self._init_modules(*args, **kwargs)
        self.save_hyperparameters()

    def _init_modules(self):
        raise NotImplementedError()

    def encode(self, x : torch.Tensor):
        mu, log_var = self.encoder(x)
        return [mu, log_var]
    
    def decode(self, z : torch.Tensor):
        result = self.decoder(z)
        return result

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        return [self.decode(z), x, mu, log_var]
    
    def generate(self, x : torch.Tensor):
        return self.forward(x)[0]
    
    def loss_function(self, x : torch.Tensor, x_recon : torch.Tensor, mu : torch.Tensor, log_var : torch.Tensor, **kwargs):
        kld_weight = kwargs['M_N']

        recons_loss = torch.nn.functional.mse_loss(x_recon, x, reduction='sum')
        kld_loss = normal_kullback_leibler_divergence(mu, log_var)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, recon, mu, log_var)
        losses = {"train_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, recon, mu, log_var)
        losses = {"val_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, recon, mu, log_var)
        losses = {"test_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['loss']
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    



class BaseVariationalBiasedAutoEncoder(BaseVAE):
    """
    Variational Context-Biased AutoEncoder
    """

    def decode(self, z: torch.Tensor, z_context: torch.Tensor):
        z = torch.cat([z, z_context], dim=1)
        return super().decode(z)

    def forward(self, x: torch.Tensor, z_context: Optional[torch.Tensor] = None):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)

        if z_context is None:
            z_context = torch.randn_like(z)
        
        x_recon = self.decode(z, z_context)
        return [x_recon, x, mu, log_var]