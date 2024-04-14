
import os
from typing import List

from ..trainer.losses import normal_kullback_leibler_divergence
from .modules.encoders import CNNVariationalEncoder
from .modules.decoders import CNNVariationalDecoder
from .modules.classifiers import MLPClassifier
from .utils import tensor_to_rgb

import torch
import pytorch_lightning as pl
import torchvision.utils as vutils


class BaseVAE(pl.LightningModule):
    """
    Modified from https://github.com/AntixK/PyTorch-VAE
    """

    def __init__(self, kld_weight: float = 0.00025, learning_rate : float = 0.005, weight_decay : float = 0.0, **kwargs):
        super(BaseVAE, self).__init__()
        self.encoder = None
        self.decoder = None

        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self._init_modules(**kwargs)
        self.save_hyperparameters()


    def _init_modules(self,**kwargs):
        raise NotImplementedError()

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("VAE")
        parser.add_argument('--kld_weight', type=float, default=0.00025, help='Weight for KLD loss.')
        parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer.')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
        return parent_parser
    

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
    
    def sample_images(self, num_samples: int = 64):
        try:
            device = next(self.parameters()).device
            # Get sample reconstruction image            
            test_input, _ = next(iter(self.trainer.datamodule.val_dataloader()))
            test_input = test_input.to(device)

            # Format image if needed
            if test_input.shape[1] > 3:
                test_input_rgb = tensor_to_rgb(test_input)
            else:
                test_input_rgb = test_input

            # Save input images
            os.makedirs(os.path.join(self.logger.log_dir, "Input_Images"), exist_ok=True)
            vutils.save_image(test_input_rgb.data,
                            os.path.join(self.logger.log_dir, 
                                        "Input_Images", 
                                        f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=12)

            # Generate reconstruction images
            recons = self.generate(test_input)
            if recons.shape[1] > 3:
                recons = tensor_to_rgb(recons)

            os.makedirs(os.path.join(self.logger.log_dir, "Reconstructions"), exist_ok=True)
            vutils.save_image(recons.data,
                            os.path.join(self.logger.log_dir, 
                                        "Reconstructions", 
                                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=12)
        
            # Generate samples
            samples = self.decode(torch.randn(num_samples,self.latent_dim).to(device))
            if samples.shape[1] > 3:
                samples = tensor_to_rgb(samples)

            os.makedirs(os.path.join(self.logger.log_dir, "Samples"), exist_ok=True)
            vutils.save_image(samples.cpu().data,
                            os.path.join(self.logger.log_dir, 
                                        "Samples",      
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=12)
        
        except StopIteration:
            pass
    
    def loss_function(self, x : torch.Tensor, x_recon : torch.Tensor, mu : torch.Tensor, log_var : torch.Tensor, **kwargs):
        kld_weight = kwargs['kld_weight'] if 'kld_weight' in kwargs else self.kld_weight

        recons_loss = torch.nn.functional.mse_loss(x_recon, x)
        kld_loss = normal_kullback_leibler_divergence(mu, log_var)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, recon, mu, log_var)
        losses = {"train_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['train_loss']
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, recon, mu, log_var)
        losses = {"val_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['val_loss']
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, recon, mu, log_var)
        losses = {"test_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['test_loss']
    
    def on_validation_end(self):
        self.sample_images()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    



class CNNVAE(BaseVAE):
    """
    Modified from https://github.com/AntixK/PyTorch-VAE
    """

    def _init_modules(self,
                 in_channels: int = 3,
                 image_dim: int = 64,
                 latent_dim: int = 256,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        
        self.latent_dim = latent_dim

        # Build Encoder
        self.encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, hidden_dims)

        # Build Decoder
        if hidden_dims is not None:
            hidden_dims.reverse()
            
        self.decoder = CNNVariationalDecoder(latent_dim, in_channels, image_dim, hidden_dims)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(CNNVAE, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("CNNVAE")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent space.')
        parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for encoder and decoder.')

        return parent_parser
    


class CNNVAEClassifier(CNNVAE):

    def _init_modules(self, 
                 in_channels: int = 3,
                 image_dim: int = 64,
                 latent_dim: int = 256,
                 hidden_dims: List = None,
                 num_classes: int = 10,
                 num_inference_layers: int = 3,
                 inference_weight: float = 1.0,
                 **kwargs):
        super()._init_modules(in_channels, image_dim, latent_dim, hidden_dims, **kwargs)

        self.inference_weight = inference_weight

        # Build Classifier
        self.classifier = MLPClassifier(latent_dim, num_classes, latent_dim, num_inference_layers)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(CNNVAEClassifier, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("CNNVarClassifier")
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--num_inference_layers', type=int, default=3, help='Number of layers in the inference network.')
        parser.add_argument('--inference_weight', type=float, default=1.0, help='Weight for inference loss.')

        return parent_parser

    def classify(self, z : torch.Tensor):
        return self.classifier(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        return [self.decode(z), self.classify(z), x, mu, log_var]
    
    def accuracy(self, y : torch.Tensor, y_recon : torch.Tensor):
        return torch.sum(y == torch.argmax(y_recon, dim=1)).float() / y.size(0)
    
    def loss_function(self, x : torch.Tensor, x_recon : torch.Tensor, y : torch.Tensor, y_recon : torch.Tensor, mu : torch.Tensor, log_var : torch.Tensor, **kwargs):
        loss = super().loss_function(x, x_recon, mu, log_var, **kwargs)

        inference_weight = kwargs['inference_weight'] if 'inference_weight' in kwargs else self.inference_weight

        inference_loss = torch.nn.functional.cross_entropy(y_recon, y)
        acc = self.accuracy(y, y_recon)

        loss["loss"] = loss["loss"] + inference_weight * inference_loss
        loss["Inference_Loss"] = inference_loss.detach()
        loss["Accuracy"] = acc.detach()
        return loss
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_recon, y_recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, x_recon, y, y_recon, mu, log_var)
        losses = {"train_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['train_loss']
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_recon, y_recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, x_recon, y, y_recon, mu, log_var)
        losses = {"val_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['val_loss']
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x_recon, y_recon, _, mu, log_var = self(x)
        losses = self.loss_function(x, x_recon, y, y_recon, mu, log_var)
        losses = {"test_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['test_loss']