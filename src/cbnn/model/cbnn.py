
from typing import Optional, List, NewType

from ..trainer.losses import normal_kullback_leibler_divergence, gaussian_mutual_information

import torch

# Custom types
VAEEncoder = torch.nn.Module
VAEDecoder = torch.nn.Module
BayesianClassifier = torch.nn.Module


class CBNN(torch.nn.Module):
    """
    Causal Bayesian Neural Network
    """

    def __init__(self, 
            context_encoder : VAEEncoder, 
            context_decoder : VAEDecoder,
            inference_encoder : VAEEncoder, 
            inference_classifier : BayesianClassifier,
            z_samples : int = 1,
            w_samples : int = 1
            ):
        super(CBNN, self).__init__()

        self.context_encoder = context_encoder
        self.context_decoder = context_decoder
        self.inference_encoder = inference_encoder
        self.inference_classifier = inference_classifier

        self.z_samples = z_samples
        self.w_samples = w_samples


    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    
    def forward(self, x: torch.Tensor, x_context: Optional[List[torch.Tensor]] = None, **kwargs):
        batch_size = x.size(0)

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
        recons_weight = kwargs['recons_weight']
        kld_weight = kwargs['kld_weight']
        context_kld_weight = kwargs['context_kld_weight']
        ic_mi_weight = kwargs['ic_mi_weight']
        wc_mi_weight = kwargs['wc_mi_weight']
        
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
