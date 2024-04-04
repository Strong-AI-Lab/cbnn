
from typing import Optional, List, NewType

from ..trainer.losses import normal_kullback_leibler_divergence, gaussian_mutual_information

import torch
import pytorch_lightning as pl

# Custom types
VAEEncoder = torch.nn.Module
VAEDecoder = torch.nn.Module
BayesianClassifier = torch.nn.Module


class CBNN(pl.LightningModule):
    """
    Causal Bayesian Neural Network
    """

    def __init__(self, 
            context_encoder : VAEEncoder, 
            context_decoder : VAEDecoder,
            inference_encoder : VAEEncoder, 
            inference_classifier : BayesianClassifier,
            z_samples : int = 1,
            w_samples : int = 1,
            nb_input_images : int = 1,
            recon_weight : float = 1.0,
            kld_weight : float = 1.0,
            context_kld_weight : float = 1.0,
            ic_mi_weight : float = 1.0,
            wc_mi_weight : float = 1.0
            ):
        super(CBNN, self).__init__()

        self.context_encoder = context_encoder
        self.context_decoder = context_decoder
        self.inference_encoder = inference_encoder
        self.inference_classifier = inference_classifier

        self.z_samples = z_samples
        self.w_samples = w_samples
        self.nb_input_images = nb_input_images

        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.context_kld_weight = context_kld_weight
        self.ic_mi_weight = ic_mi_weight
        self.wc_mi_weight = wc_mi_weight

        self.save_hyperparameters(ignore='context_encoder,context_decoder,inference_encoder,inference_classifier')


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
