
from typing import Optional, List, NewType

from .modules.encoders import CNNVariationalEncoder
from .modules.decoders import CNNVariationalDecoder
from .modules.classifiers import BayesianClassifier
from ..trainer.losses import normal_kullback_leibler_divergence, gaussian_mutual_information

import torch
import pytorch_lightning as pl


class CBNN(pl.LightningModule):
    """
    Causal Bayesian Neural Network
    """

    def __init__(self,
            z_samples : int = 1,
            w_samples : int = 1,
            nb_input_images : int = 1,
            recon_weight : float = 1.0,
            kld_weight : float = 1.0,
            context_kld_weight : float = 1.0,
            ic_mi_weight : float = 1.0,
            wc_mi_weight : float = 1.0,
            **kwargs
            ):
        super(CBNN, self).__init__()

        # Modules
        self.context_encoder = None
        self.context_decoder = None
        self.inference_encoder = None
        self.inference_classifier = None

        # Sampling parameters
        self.z_samples = z_samples
        self.w_samples = w_samples
        self.nb_input_images = nb_input_images

        # Loss weights
        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.context_kld_weight = context_kld_weight
        self.ic_mi_weight = ic_mi_weight
        self.wc_mi_weight = wc_mi_weight

        self._init_modules(**kwargs)

        self.save_hyperparameters()


    def _init_modules(self,**kwargs):
        raise NotImplementedError()

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        
        parser = parent_parser.add_argument_group("CBNN")
        parser.add_argument('--z_samples', type=int, default=1, help='Number of samples to draw from the inference encoder.')
        parser.add_argument('--w_samples', type=int, default=1, help='Number of weight samples to draw from the meta inference classifier.')
        parser.add_argument('--nb_input_images', type=int, default=1, help='Number of input images to the model.')
        parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight of the reconstruction loss.')
        parser.add_argument('--kld_weight', type=float, default=1.0, help='Weight of the Kullback-Leibler divergence loss.')
        parser.add_argument('--context_kld_weight', type=float, default=1.0, help='Weight of the context Kullback-Leibler divergence loss.')
        parser.add_argument('--ic_mi_weight', type=float, default=1.0, help='Weight of the Inference-Context Mutual Information loss.')
        parser.add_argument('--wc_mi_weight', type=float, default=1.0, help='Weight of the Weights-Context Mutual Information loss.')

        return parent_parser


    def pre_load_context(self, x_context: List[torch.Tensor]):
        self.loaded_context = True
        self.x_context = x_context

    def clear_context(self):
        self.loaded_context = False
        self.x_context = None


    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    
    def forward(self, x: torch.Tensor, x_context: Optional[List[torch.Tensor]] = None, **kwargs):
        batch_size = x.size(0)

        if self.nb_input_images > 1:
            assert len(x.size()) > 4, "Input tensor must have at least 5 dimensions when nb_input_images is > 1: (batch, nb_input_images, channels, height, width)"
            assert x.size(-4) == self.nb_input_images, "Input tensor must have the same number of images as specified by nb_input_images"

            x = x.view(-1, *x.size()[-3:])
            if x_context is not None:
                x_context = [x_c.view(-1, *x_c.size()[-3:]) for x_c in x_context]

        if x_context is None:
            context_mu, context_log_var = self.context_encoder(x)
        else:
            context_mus = []
            context_log_vars = []
            for x_c in x_context:
                mu_c, log_var_c = self.context_encoder(x_c)
                context_mus.append(mu_c)
                context_log_vars.append(log_var_c)

        mu, log_var = self.inference_encoder(x)

        if self.nb_input_images > 1:
            mu = mu.view(batch_size, self.nb_input_images * mu.size(-1))
            log_var = log_var.view(batch_size, self.nb_input_images * log_var.size(-1))

            if x_context is not None:
                context_mu = context_mu.view(batch_size, self.nb_input_images * context_mu.size(-1))
                context_log_var = context_log_var.view(batch_size, self.nb_input_images * context_log_var.size(-1))
            else:
                context_mus = [mu.view(batch_size, self.nb_input_images * mu.size(-1)) for mu in context_mus]
                context_log_vars = [log_var.view(batch_size, self.nb_input_images * log_var.size(-1)) for log_var in context_log_vars]

        x_recons = []
        ys = []
        ws = []
        context_zs = []
        zs = []
        for i in range(self.z_samples):
            if x_context is not None:
                context_mu = context_mus[i]
                context_log_var = context_log_vars[i]

            context_z = self.sample(context_mu, context_log_var)
            z = self.sample(mu, log_var)
            context_zs.append(context_z)
            zs.append(z)

            x_recons.append(self.context_decoder(context_z))

            for _ in range(self.w_samples):
                y_j, *w_j = self.inference_classifier(z, context_z)
                ys.append(y_j)
                ws.append(torch.cat([w.view(batch_size,-1) for w in w_j], dim=0))

        y = torch.stack(ys).mean(dim=0)

        outputs = [x_recons, y, ws, zs, context_zs, mu, log_var]
        if x_context is None:
            outputs.extend([[context_mu], [context_log_var]])
        else:
            outputs.extend([context_mus, context_log_vars])

        return outputs
    
    
    def loss_function(self, 
            x : torch.Tensor, 
            x_recons : List[torch.Tensor], 
            y_target : torch.Tensor, 
            y_recon : torch.Tensor,
            weight_samples : List[torch.Tensor],
            z_samples : List[torch.Tensor],
            context_z_samples : List[torch.Tensor],
            mu : torch.Tensor, 
            log_var : torch.Tensor, 
            context_mus : List[torch.Tensor], 
            context_log_vars : List[torch.Tensor], 
            **kwargs
            ):
        recons_weight = kwargs['recons_weight'] if 'recons_weight' in kwargs else self.recon_weight
        kld_weight = kwargs['kld_weight'] if 'kld_weight' in kwargs else self.kld_weight
        context_kld_weight = kwargs['context_kld_weight'] if 'context_kld_weight' in kwargs else self.context_kld_weight
        ic_mi_weight = kwargs['ic_mi_weight'] if 'ic_mi_weight' in kwargs else self.ic_mi_weight
        wc_mi_weight = kwargs['wc_mi_weight'] if 'wc_mi_weight' in kwargs else self.wc_mi_weight
        
        # Main inference loss
        infer_loss = torch.nn.functional.cross_entropy(y_recon, y_target, reduction='sum')

        # Context reconstruction loss
        recons_loss = 0.0
        for x_recon in x_recons:
            recons_loss += torch.nn.functional.mse_loss(x_recon, x, reduction='sum')
        recons_loss /= len(x_recons)

        # Context Encoder Kullback-Leibler divergence loss
        context_kld_loss = 0.0
        for context_mu, context_log_var in zip(context_mus, context_log_vars):
            context_kld_loss += normal_kullback_leibler_divergence(context_mu, context_log_var)
        context_kld_loss /= len(context_mus)

        # Inference Encoder Kullback-Leibler divergence loss
        kld_loss = normal_kullback_leibler_divergence(mu, log_var)

        # Inference-Context Mutual Information loss
        ic_mi_loss = gaussian_mutual_information(z_samples, context_z_samples)

        # Weights-Context Mutual Information loss
        repeated_context_z_samples = [z for z in context_z_samples for _ in range(self.w_samples)]
        wc_mi_loss = gaussian_mutual_information(weight_samples, repeated_context_z_samples)


        loss = infer_loss + recons_weight * recons_loss + context_kld_weight * context_kld_loss + kld_weight * kld_loss + ic_mi_weight * ic_mi_loss + wc_mi_weight * wc_mi_loss
        return {'loss': loss, 
                'Inference_Loss':infer_loss.detach(), 
                'Reconstruction_Loss':recons_loss.detach(), 
                'Context_KLD':-context_kld_loss.detach(),
                'KLD':-kld_loss.detach(), 
                'IC_MI':ic_mi_loss.detach(), 
                'WC_MI':wc_mi_loss.detach()
                }
    

    def training_step(self, batch, batch_idx):
        if self.loaded_context:
            x_context = self.x_context
        else:
            x_context = None

        x, y = batch
        x_recons, y_recon, *outputs = self(x, x_context)

        losses = self.loss_function(x, x_recons, y, y_recon, *outputs)
        losses = {"train_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        if self.loaded_context:
            x_context = self.x_context
        else:
            x_context = None

        x, y = batch
        x_recons, y_recon, *outputs = self(x, x_context)
        
        losses = self.loss_function(x, x_recons, y, y_recon, *outputs)
        losses = {"val_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        if self.loaded_context:
            x_context = self.x_context
        else:
            x_context = None

        x, y = batch
        x_recons, y_recon, *outputs = self(x, x_context)
        
        losses = self.loss_function(x, x_recons, y, y_recon, *outputs)
        losses = {"test_" + k: v for k, v in losses.items()}
        self.log_dict(losses)
        return losses['loss']
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
    



class CNN_CBNN(CBNN):

    def _init_modules(self,
            in_channels: int = 3,
            out_channels: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None, 
            classifier_hidden_dim: int = 128, 
            classifier_nb_layers: int = 3,
            **kwargs):
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(latent_dim, in_channels, encoder_hidden_dims.reverse() if encoder_hidden_dims is not None else None)
        self.inference_encoder = CNNVariationalEncoder(in_channels, latent_dim, encoder_hidden_dims)
        self.inference_classifier = BayesianClassifier(2 * latent_dim, out_channels, classifier_hidden_dim, classifier_nb_layers)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(CNN_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("CNN_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--out_channels', type=int, default=10, help='Number of output channels.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        parser.add_argument('--classifier_hidden_dim', type=int, default=128, help='Hidden dimension for the classifier.')
        parser.add_argument('--classifier_nb_layers', type=int, default=3, help='Number of layers for the classifier.')
        
        return parent_parser