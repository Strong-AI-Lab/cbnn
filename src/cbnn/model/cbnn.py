
from typing import Optional, List

from .modules.encoders import CNNVariationalEncoder
from .modules.decoders import CNNVariationalDecoder
from .modules.classifiers import BayesianClassifier, MCQABayesClassifier
from .resnet import ResNet18
from .modules.invariant_resnet_utils import resnet18_invariant
from .modules.bayes_resnet_utils import bayes_resnet18_invariant, mcqa_bayes_resnet18_invariant, mipred_bayes_resnet18_invariant
from ..trainer.losses import normal_kullback_leibler_divergence, gaussian_mutual_information
from .utils import sample_images, INFERENCE_CONTEXT_COLLATORS

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
            w_kld_weight : float = 1.0,
            ic_mi_weight : float = 1.0,
            wc_mi_weight : float = 1.0,
            learning_rate : float = 0.005, 
            weight_decay : float = 0.0,
            l2_regularisation : float = 0.0,
            context_inference_weight : float = 0.0,
            inverse_context : bool = False,
            inference_without_encoder : bool = False,
            sample_context_from_distribution : bool = False,
            freeze_parameters : Optional[List[str]] = None,
            reverse_freeze : bool = False,
            split_recons_infer_latents : Optional[float] = None,
            context_split_mi_weight : float = 1.0,
            inference_context_collator : str = "cat",
            weight_pseudo_riemann_regularisation : float = 1.0,
            **kwargs
            ):
        super(CBNN, self).__init__()

        # Modules
        self.context_encoder = None
        self.context_decoder = None
        self.inference_encoder = None
        self.inference_classifier = None
        self.x_context = None
        self.y_context = None

        # Sampling parameters
        self.z_samples = z_samples
        self.w_samples = w_samples
        self.nb_input_images = nb_input_images
        self.inverse_context = inverse_context
        self.inference_without_encoder = inference_without_encoder
        self.sample_context_from_distribution = sample_context_from_distribution
        self.split_recons_infer_latents = split_recons_infer_latents
        self.latent_dim = None

        # Loss weights
        self.recon_weight = recon_weight
        self.kld_weight = kld_weight
        self.context_kld_weight = context_kld_weight
        self.w_kld_weight = w_kld_weight
        self.ic_mi_weight = ic_mi_weight
        self.wc_mi_weight = wc_mi_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l2_regularisation = l2_regularisation
        self.context_inference_weight = context_inference_weight
        self.context_split_mi_weight = context_split_mi_weight
        self.weight_pseudo_riemann_regularisation = weight_pseudo_riemann_regularisation

        # Context loader
        self.context_distrib_loader_funcs = {
            "train" : None,
            "val" : None,
            "test" : None
        }

        # Context collator
        self.inference_context_collator = INFERENCE_CONTEXT_COLLATORS[inference_context_collator]

        self._init_modules(**kwargs)

        # Freeze parameters
        if freeze_parameters is not None:
            self.partial_freeze(freeze_parameters, reverse=reverse_freeze)

        self.save_hyperparameters()


    def _init_modules(self,**kwargs): # at least self.latent_dim, must be initialised in the child class
        raise NotImplementedError()
    
    @property
    def recons_latent_dim(self):
        if self.split_recons_infer_latents is not None:
            return int(self.latent_dim * self.split_recons_infer_latents)
        else:
            return self.latent_dim
        
    @property
    def infer_latent_dim(self):
        if self.split_recons_infer_latents is not None:
            return self.latent_dim - int(self.latent_dim * self.split_recons_infer_latents)
        else:
            return self.latent_dim

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        
        parser = parent_parser.add_argument_group("CBNN")
        parser.add_argument('--z_samples', type=int, default=1, help='Number of samples to draw from the inference encoder.')
        parser.add_argument('--w_samples', type=int, default=1, help='Number of weight samples to draw from the meta inference classifier.')
        parser.add_argument('--nb_input_images', type=int, default=1, help='Number of input images to the model.')
        parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight of the reconstruction loss.')
        parser.add_argument('--kld_weight', type=float, default=1.0, help='Weight of the Kullback-Leibler divergence loss.')
        parser.add_argument('--context_kld_weight', type=float, default=1.0, help='Weight of the context Kullback-Leibler divergence loss.')
        parser.add_argument('--w_kld_weight', type=float, default=1.0, help='Weight of the weights Kullback-Leibler divergence loss.')
        parser.add_argument('--ic_mi_weight', type=float, default=1.0, help='Weight of the Inference-Context Mutual Information loss.')
        parser.add_argument('--wc_mi_weight', type=float, default=1.0, help='Weight of the Weights-Context Mutual Information loss.')
        parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer.')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
        parser.add_argument('--l2_regularisation', type=float, default=0.0, help='Additional L2 regularisation for the sampled weights.')
        parser.add_argument('--context_inference_weight', type=float, default=0.4, help='Weight of the context inference labels in the mixed labels.')
        parser.add_argument('--inverse_context', action='store_true', help='Switch context and inference encoders to be reflect the theoretical groundings.')
        parser.add_argument('--inference_without_encoder', action='store_true', help='Use when the inference network is a single decoder module and does not require an encoder.')
        parser.add_argument('--sample_context_from_distribution', action='store_true', help='Sample context from the distribution. If false, use the input as context')
        parser.add_argument('--freeze_parameters', type=str, nargs='+', default=None, help='List of parameters to freeze during training.')
        parser.add_argument('--reverse_freeze', action='store_true', help='Reverse the freeze list to freeze all parameters except those in the list.')
        parser.add_argument('--split_recons_infer_latents', type=Optional[float], default=None, help='Split the latent space into two parts for reconstruction and inference.')
        parser.add_argument('--context_split_mi_weight', type=float, default=1.0, help='Weight of the context split mutual information loss.')
        parser.add_argument('--weight_pseudo_riemann_regularisation', type=float, default=1.0, help='Pseudo-Riemannian regularisation for the weight space.')

        return parent_parser

    @property
    def loaded_context(self):
        return self.x_context is not None

    def pre_load_context(self, x_context: List[torch.Tensor], y_context: Optional[List[torch.Tensor]] = None):
        self.x_context = x_context
        self.y_context = y_context

    def clear_context(self):
        self.x_context = None
        self.y_context = None


    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var * 0.001 * 1/torch.sqrt(torch.tensor(log_var.size(-1)).to(log_var.device)))
        eps = torch.randn_like(std)
        return mu + eps*std

    
    def forward(self, x: torch.Tensor, x_context: Optional[List[torch.Tensor]] = None, **kwargs):
        x_shape = x.size()
        batch_size = x_shape[0]

        if x_context is None: # use input as context
            x_context = [x]
        else: # adjust batch size mismatch
            c_batch_size = x_context[0].size(0)
            if c_batch_size > batch_size:
                x_context = [x_c[:batch_size] for x_c in x_context]
            elif c_batch_size < batch_size:
                x_context = [torch.cat([x_c, x_c[:batch_size-c_batch_size]]) for x_c in x_context]

        if self.inference_without_encoder: # if inference network is a single decoder module, do not use encoder and create dummy encoder instead
            inference_encoding_func = lambda x: (torch.zeros(x.size(0), 1), torch.ones(x.size(0), 1))
        else:
            inference_encoding_func = self.inference_encoder

        if self.inverse_context: # if inverse context, switch context and inference encoders to reconstruct inference input instead of context
            context_encoder = inference_encoding_func
            inference_encoder = self.context_encoder
        else:
            context_encoder = self.context_encoder
            inference_encoder = inference_encoding_func


        if self.nb_input_images > 1: # if multiple input images, reshape input tensor to (batch * nb_input_images, channels, height, width)
            assert len(x.size()) > 4, f"Input tensor must have at least 5 dimensions when nb_input_images is > 1: (batch, nb_input_images, channels, height, width). Got {len(x.size())} dimensions: {x.size()}"
            assert x.size(-4) == self.nb_input_images, f"Input tensor must have the same number of images as specified by nb_input_images. Got {x.size(-4)} images ({x.size()}), expected {self.nb_input_images}."

            x = x.view(-1, *x.size()[-3:])
            x_context = [x_c.view(-1, *x_c.size()[-3:]) for x_c in x_context]
        
        x_c = torch.cat(x_context) # concat context inputs at batch size for the encoding step

        # Encode context
        mu_c, log_var_c = context_encoder(x_c)
        context_mus = mu_c.chunk(self.z_samples, dim=0)
        context_log_vars = log_var_c.chunk(self.z_samples, dim=0)

        # Encode inference input
        inference_mu, inference_log_var = inference_encoder(x)

        if self.nb_input_images > 1: # if multiple input images, reshape inference latent variables to (batch, nb_input_images * latent_dim)
            inference_mu = inference_mu.view(batch_size, self.nb_input_images * inference_mu.size(-1))
            inference_log_var = inference_log_var.view(batch_size, self.nb_input_images * inference_log_var.size(-1))

            context_mus = [c_mu.view(batch_size, self.nb_input_images * c_mu.size(-1)) for c_mu in context_mus]
            context_log_vars = [c_log_var.view(batch_size, self.nb_input_images * c_log_var.size(-1)) for c_log_var in context_log_vars]

        x_recons = []
        context_zs = []
        zs = []
        infer_context_zs = []
        infer_zs = []
        for i in range(self.z_samples): # sample z from context and inference latent variables
            c_idx = i if i < len(context_mus) else -1
            if self.inverse_context:
                mu = context_mus[c_idx]
                log_var = context_log_vars[c_idx]
                context_mu = inference_mu
                context_log_var = inference_log_var
            else:
                mu = inference_mu
                log_var = inference_log_var
                context_mu = context_mus[c_idx]
                context_log_var = context_log_vars[c_idx]


            if self.inference_without_encoder: # mu, log_var do not exist (values are not filled)
                if self.inverse_context:
                    z = x_context[c_idx].view(x_shape)
                else:
                    z = x.view(x_shape)
                context_z = self.sample(context_mu, context_log_var)
            else:
                context_z = self.sample(context_mu, context_log_var)
                z = self.sample(mu, log_var)


            context_zs.append(context_z)
            zs.append(z)

            
            if self.split_recons_infer_latents is not None: # If option is selected, split latent space for reconstruction and inference (extract first part for reconstruction)
                recon_context_z = context_z[..., :int(self.split_recons_infer_latents * context_z.size(-1))]
            else:
                recon_context_z = context_z

            if self.recon_weight > 0.0: # Perform reconstruction
                if self.nb_input_images > 1:
                    recon_context_z = recon_context_z.view(batch_size * self.nb_input_images, -1)
                    x_recon = self.context_decoder(recon_context_z)
                    x_recon = x_recon.view(batch_size, self.nb_input_images, *x_recon.size()[1:])
                else:
                    x_recon = self.context_decoder(recon_context_z)

                x_recons.append(x_recon)

            if self.split_recons_infer_latents is not None: # If option is selected, split latent space for reconstruction and inference (extract second part for inference)
                infer_context_z = context_z[..., int(self.split_recons_infer_latents * context_z.size(-1)):]
            else:
                infer_context_z = context_z

            infer_context_zs.append(infer_context_z)
            infer_zs.append(z)


        ys = []
        ws = []
        for _ in range(self.w_samples): # sample weights from the classifier and perform inference
            y_j, *w_j = self.inference_context_collator(infer_zs, infer_context_zs, self.inference_classifier)
            ys.append(y_j)
            ws.append(torch.cat([w.view(-1) for w in w_j], dim=0))

        y = torch.stack(ys).mean(dim=0)

        outputs = [x_recons, y, ys, ws, zs, context_zs, inference_mu, inference_log_var]
        outputs.extend([context_mus, context_log_vars])

        return outputs
    

    def generate(self, x: torch.Tensor):
        mu, log_var = self.context_encoder(x)
        z = self.sample(mu, log_var)

        if self.split_recons_infer_latents is not None:
            z = z[..., :int(self.split_recons_infer_latents * z.size(-1))]

        x_recon = self.context_decoder(z)
        return x_recon
    
    def decode(self, z: torch.Tensor):
        x_recon = self.context_decoder(z)
        return x_recon
    
    def sample_images(self, num_samples: int = 64):
        inp, _ = next(iter(self.trainer.datamodule.val_dataloader()))
        if self.nb_input_images > 1:
            inp = inp.view(-1, *inp.size()[2:])
        sample_images(self, inp, self.logger.log_dir, self.logger.name, self.current_epoch, num_samples)

    
    
    def accuracy(self, y_recon : torch.Tensor, y : torch.Tensor):
        if len(y.size()) > 1: # y is a one-hot encoded tensor or a probability distribution, convert to maximum likelihood class labels
            y = torch.argmax(y, dim=1)
        return torch.sum(y == torch.argmax(y_recon, dim=1)).float() / y.size(0)
    
    def loss_function(self, 
            x : torch.Tensor, 
            x_recons : List[torch.Tensor], 
            y_target : torch.Tensor, 
            y_recon : torch.Tensor,
            y_samples : List[torch.Tensor],
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
        w_kld_weight = kwargs['w_kld_weight'] if 'w_kld_weight' in kwargs else self.w_kld_weight
        ic_mi_weight = kwargs['ic_mi_weight'] if 'ic_mi_weight' in kwargs else self.ic_mi_weight
        wc_mi_weight = kwargs['wc_mi_weight'] if 'wc_mi_weight' in kwargs else self.wc_mi_weight
        context_split_mi_weight = kwargs['context_split_mi_weight'] if 'context_split_mi_weight' in kwargs else self.context_split_mi_weight
        l2_regularisation = kwargs['l2_regularisation'] if 'l2_regularisation' in kwargs else self.l2_regularisation
        weight_pseudo_riemann_regularisation = kwargs['weight_pseudo_riemann_regularisation'] if 'weight_pseudo_riemann_regularisation' in kwargs else self.weight_pseudo_riemann_regularisation
        
        # Main inference loss
        infer_loss = torch.nn.functional.cross_entropy(y_recon, y_target)

        # Context reconstruction loss
        recons_loss = torch.tensor(0.0).to(x.device)
        if self.recon_weight > 0.0:
            for x_recon in x_recons:
                recons_loss += torch.nn.functional.mse_loss(x_recon, x)
            recons_loss /= len(x_recons)


        # Context Encoder Kullback-Leibler divergence loss and variance
        context_kld_loss = torch.tensor(0.0).to(x.device)
        context_kld_var = torch.tensor(0.0).to(x.device)
        if context_kld_weight > 0.0:
            for context_mu, context_log_var in zip(context_mus, context_log_vars):
                context_kld_loss += normal_kullback_leibler_divergence(context_mu, context_log_var)
                context_kld_var += torch.exp(context_log_var).mean()
            context_kld_loss /= len(context_mus)
            context_kld_var /= len(context_log_vars)

        # Inference Encoder Kullback-Leibler divergence loss and variance
        if kld_weight > 0.0:
            kld_loss = normal_kullback_leibler_divergence(mu, log_var)
            kld_var = torch.exp(log_var).mean()
        else:
            kld_loss = torch.tensor(0.0).to(x.device)
            kld_var = torch.tensor(0.0).to(x.device)

        # Weights Kullback-Leibler divergence loss and variance
        if w_kld_weight > 0.0:
            weights_mean, weights_log_var = self.inference_classifier.get_weight_distributions()
            w_kld_loss = normal_kullback_leibler_divergence(weights_mean, weights_log_var)
            w_kld_var = torch.exp(weights_log_var).mean()
        else:
            w_kld_loss = torch.tensor(0.0).to(x.device)
            w_kld_var = torch.tensor(0.0).to(x.device)

        # Latent split Mutual Information loss
        context_split_mi_loss = torch.tensor(0.0).to(x.device)
        if self.split_recons_infer_latents is not None:
            context_z = torch.cat(context_z_samples, dim=0)
            recons_context_z = context_z[..., :int(self.split_recons_infer_latents * context_z.size(-1))]
            infer_context_z = context_z[..., int(self.split_recons_infer_latents * context_z.size(-1)):]
            context_split_mi_loss = gaussian_mutual_information(recons_context_z, infer_context_z, top_k=16)

        # Inference-Context Mutual Information loss
        ic_mi_loss = torch.tensor(0.0).to(x.device)
        if ic_mi_weight > 0.0:
            ic_mi_loss = gaussian_mutual_information(torch.cat(z_samples, dim=0), torch.cat(context_z_samples, dim=0),  top_k=16)

        # Weights-Context Mutual Information loss
        wc_mi_loss = torch.tensor(0.0).to(x.device)
        if wc_mi_weight > 0.0:
            repeated_w = torch.cat([w.view(1,-1).repeat(context_z_samples[0].size(0) // self.w_samples, 1) for w in weight_samples], dim=0)
            context_z = torch.cat(context_z_samples, dim=0)
            wc_mi_loss = gaussian_mutual_information(repeated_w, context_z, max_dim=128, top_k=16)

        # L2 weight regularisation
        l2_loss = torch.tensor(0.0).to(x.device)
        if l2_regularisation > 0.0:
            l2_loss = torch.cat([w.view(-1) for w in weight_samples]).pow(2).sum() / (2 * len(weight_samples))

        # Pseudo-Riemann regularisation
        pseudo_riemann_loss = torch.tensor(0.0).to(x.device)
        if weight_pseudo_riemann_regularisation > 0.0 and len(y_samples) > 1:
            for i, y_i in enumerate(y_samples):
                for y_j in y_samples[i+1:]:
                    pseudo_riemann_loss += torch.nn.functional.mse_loss(y_i, y_j)
            pseudo_riemann_loss /= len(y_samples) * (len(y_samples) - 1) / 2


        loss = infer_loss + recons_weight * recons_loss + context_kld_weight * context_kld_loss + kld_weight * kld_loss + w_kld_weight * w_kld_loss + context_split_mi_weight * context_split_mi_loss + ic_mi_weight * ic_mi_loss + wc_mi_weight * wc_mi_loss + self.l2_regularisation * l2_loss + weight_pseudo_riemann_regularisation * pseudo_riemann_loss
        return {'loss': loss, 
                'Inference_Loss':infer_loss.detach(), 
                'Reconstruction_Loss':recons_loss.detach(), 
                'Context_KLD':-context_kld_loss.detach(),
                'Weights_KLD':-w_kld_loss.detach(),
                'KLD':-kld_loss.detach(), 
                'Context_Split_MI':context_split_mi_loss.detach(),
                'IC_MI':ic_mi_loss.detach(), 
                'WC_MI':wc_mi_loss.detach(),
                'L2': l2_loss.detach(),
                'Accuracy':self.accuracy(y_recon, y_target).detach(),
                'Context_KLD_Var':context_kld_var.detach(),
                'Weights_KLD_Var':w_kld_var.detach(),
                'KLD_Var':kld_var.detach(),
                'Pseudo_Riemann':pseudo_riemann_loss.detach()
                }
    
    def _mix_labels(self, y_target : torch.Tensor, labels_to_mix : List[torch.Tensor], num_classes : int):
        y_mix = torch.nn.functional.one_hot(y_target, num_classes=num_classes) * (1 - self.context_inference_weight)
        for l_i in labels_to_mix:
            y_mix += torch.nn.functional.one_hot(l_i, num_classes=num_classes) * self.context_inference_weight / len(labels_to_mix)
        return y_mix

    def _sample_context_from_distribution(self, split : str = "train", label : Optional[torch.Tensor] = None):
        dataloader = self.context_distrib_loader_funcs[split]()
        context_list = []
        context_label_list = []

        if label is None:
            for _ in range(self.z_samples):
                x_context, y_context = next(iter(dataloader))
                context_list.append(x_context.to(self.device))
                context_label_list.append(y_context.to(self.device))
        
        else: # match context to label
            candidates = {i:[] for i in label.unique().tolist()}
            for _ in range(self.z_samples):
                while not all([len(candidates[i]) >= (label==i).sum().item() for i in candidates]):
                    x_context, y_context = next(iter(dataloader))
                    for i in range(len(y_context)):
                        if y_context[i].item() in candidates:
                            candidates[y_context[i].item()].append(x_context[i])
                selected_context = []
                for i in label.tolist():
                    selected_context.append(candidates[i].pop(0))
                context_list.append(torch.stack(selected_context).to(self.device))
                context_label_list.append(label.to(self.device))
        
        self.pre_load_context(context_list, context_label_list)

    def _fill_context_dataloader(self):
        self.context_distrib_loader_funcs["train"] = self.trainer.datamodule.train_dataloader
        self.context_distrib_loader_funcs["val"] = self.trainer.datamodule.val_dataloader
        self.context_distrib_loader_funcs["test"] = self.trainer.datamodule.test_dataloader

    def on_fit_start(self):
        self._fill_context_dataloader()

    def on_test_start(self):
        self._fill_context_dataloader()
    
    def on_predict_start(self):
        self._fill_context_dataloader()


    def training_step(self, batch, batch_idx):
        if self.sample_context_from_distribution:
            self._sample_context_from_distribution(split="train")

        x_context = self.x_context
        y_context = self.y_context

        x, y = batch
        x_recons, y_recon, *outputs = self(x, x_context)
        if self.context_inference_weight > 0.0:
            y = self._mix_labels(y, y_context, y_recon.size(-1))

        losses = self.loss_function(x, x_recons, y, y_recon, *outputs)
        losses = {"train_" + k: v for k, v in losses.items()}
        self.log_dict(losses, sync_dist=True)
        return losses['train_loss']
    
    def validation_step(self, batch, batch_idx):
        if self.sample_context_from_distribution:
            self._sample_context_from_distribution(split="val")

        x_context = self.x_context

        x, y = batch
        x_recons, y_recon, *outputs = self(x, x_context)
        
        losses = self.loss_function(x, x_recons, y, y_recon, *outputs)
        losses = {"val_" + k: v for k, v in losses.items()}
        self.log_dict(losses, sync_dist=True)
        return losses['val_loss']
    
    def on_validation_end(self):
        self.sample_images()
    
    def test_step(self, batch, batch_idx):
        if self.sample_context_from_distribution:
            self._sample_context_from_distribution(split="test")

        x_context = self.x_context

        x, y = batch
        x_recons, y_recon, *outputs = self(x, x_context)
        
        losses = self.loss_function(x, x_recons, y, y_recon, *outputs)
        losses = {"test_" + k: v for k, v in losses.items()}
        self.log_dict(losses, sync_dist=True)
        return losses['test_loss']
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def partial_freeze(self, frozen_layers : List[str], reverse : bool = False):
        for name, param in self.named_parameters():
            if (name in frozen_layers and not reverse) or (name not in frozen_layers and reverse):
                param.requires_grad_(False)
        
    



class CNN_CBNN(CBNN):

    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            out_channels: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None, 
            classifier_hidden_dim: int = 128, 
            classifier_nb_layers: int = 3,
            **kwargs):
        
        self.latent_dim = latent_dim
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, encoder_hidden_dims[::-1] if encoder_hidden_dims is not None else None)
        self.inference_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.inference_classifier = BayesianClassifier(2 * self.nb_input_images * self.infer_latent_dim, out_channels, classifier_hidden_dim, classifier_nb_layers)

    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(CNN_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("CNN_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--out_channels', type=int, default=10, help='Number of output channels.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        parser.add_argument('--classifier_hidden_dim', type=int, default=128, help='Hidden dimension for the classifier.')
        parser.add_argument('--classifier_nb_layers', type=int, default=3, help='Number of layers for the classifier.')
        
        return parent_parser
    



class MCQA_CNN_CBNN(CBNN):

    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            out_channels: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None, 
            classifier_hidden_dim: int = 128, 
            classifier_nb_layers: int = 3,
            **kwargs):
        
        self.latent_dim = latent_dim
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, encoder_hidden_dims[::-1] if encoder_hidden_dims is not None else None)
        self.inference_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.inference_classifier = MCQABayesClassifier(2 * self.nb_input_images * self.infer_latent_dim, classifier_hidden_dim, classifier_nb_layers, self.nb_input_images-out_channels, out_channels)

    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MCQA_CNN_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("CNN_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--out_channels', type=int, default=10, help='Number of output channels.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        parser.add_argument('--classifier_hidden_dim', type=int, default=128, help='Hidden dimension for the classifier.')
        parser.add_argument('--classifier_nb_layers', type=int, default=3, help='Number of layers for the classifier.')
        
        return parent_parser
    



class ResNet_CBNN(CBNN):
    def __init__(self, **kwargs):
        if 'inference_without_encoder' in kwargs:
            kwargs.pop('inference_without_encoder')
        if 'inference_context_collator' in kwargs:
            kwargs.pop('inference_context_collator')
        super(ResNet_CBNN, self).__init__(inference_without_encoder=True, inference_context_collator='none', **kwargs)


    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            num_classes: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None,
            **kwargs):
        
        self.latent_dim = latent_dim
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, encoder_hidden_dims[::-1] if encoder_hidden_dims is not None else None)
        self.inference_encoder = None
        self.inference_classifier = bayes_resnet18_invariant(in_channels, num_classes, self.infer_latent_dim, image_size=image_dim)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(ResNet_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("ResNet_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        
        return parent_parser
    
class ResNet_CT(CBNN):
    def __init__(self, **kwargs):
        if 'inference_without_encoder' in kwargs:
            kwargs.pop('inference_without_encoder')
        if 'inference_context_collator' in kwargs:
            kwargs.pop('inference_context_collator')
        super(ResNet_CT, self).__init__(inference_without_encoder=True, inference_context_collator='none', **kwargs)


    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            num_classes: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None,
            **kwargs):
        
        self.latent_dim = latent_dim
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, encoder_hidden_dims[::-1] if encoder_hidden_dims is not None else None)
        self.inference_encoder = None
        self.inference_classifier = resnet18_invariant(in_channels=in_channels, num_classes=num_classes, invariant_dim=latent_dim, image_size=image_dim)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(ResNet_CT, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("ResNet_CT")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        
        return parent_parser




class MCQA_ResNet_CBNN(CBNN):
    def __init__(self, **kwargs):
        if 'inference_without_encoder' in kwargs:
            kwargs.pop('inference_without_encoder')
        if 'inference_context_collator' in kwargs:
            kwargs.pop('inference_context_collator')
        super(ResNet_CBNN, self).__init__(inference_without_encoder=True, inference_context_collator='none', **kwargs)


    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            num_classes: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None,
            **kwargs):
        
        self.latent_dim = latent_dim
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, encoder_hidden_dims[::-1] if encoder_hidden_dims is not None else None)
        self.inference_encoder = None
        self.inference_classifier = mcqa_bayes_resnet18_invariant(in_channels, self.nb_input_images, self.infer_latent_dim, num_classes, self.infer_latent_dim, image_size=image_dim)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MCQA_ResNet_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("MCQA_ResNet_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        
        return parent_parser




class MIPred_ResNet_CBNN(CBNN):
    def __init__(self, **kwargs):
        if 'inference_without_encoder' in kwargs:
            kwargs.pop('inference_without_encoder')
        if 'inference_context_collator' in kwargs:
            kwargs.pop('inference_context_collator')
        super(ResNet_CBNN, self).__init__(inference_without_encoder=True, inference_context_collator='none', **kwargs)


    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            num_classes: int = 10,
            latent_dim: int = 256,
            encoder_hidden_dims: List = None,
            **kwargs):
        
        self.latent_dim = latent_dim
        
        # Build modules
        self.context_encoder = CNNVariationalEncoder(in_channels, image_dim, latent_dim, encoder_hidden_dims)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, encoder_hidden_dims[::-1] if encoder_hidden_dims is not None else None)
        self.inference_encoder = None
        self.inference_classifier = mipred_bayes_resnet18_invariant(in_channels, self.nb_input_images, self.infer_latent_dim, num_classes, self.infer_latent_dim, image_size=image_dim)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MIPred_ResNet_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("MIPred_ResNet_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--encoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the encoder.')
        
        return parent_parser




class ResEnc_CBNN(CBNN):
    class ResNetEncoder(torch.nn.Module):
        def __init__(self, resnet : torch.nn.Module, latent_dim : int = 256):
            super(ResEnc_CBNN.ResNetEncoder, self).__init__()
            self.resnet_encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            self.resnet_encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            last_block =  torch.nn.Sequential(*list(self.resnet_encoder[-2][-1].children()))
            
            old_conv2 = last_block[-2]
            old_bn2 = last_block[-1]
            last_block[-2] = torch.nn.Conv2d(old_conv2.in_channels, 2*latent_dim, kernel_size=old_conv2.kernel_size, stride=old_conv2.stride, padding=old_conv2.padding, bias=old_conv2.bias is not None)
            last_block[-1] = type(old_bn2)(2*latent_dim, eps=old_bn2.eps, momentum=old_bn2.momentum, affine=old_bn2.affine, track_running_stats=old_bn2.track_running_stats)
            self.resnet_encoder[-2][-1] = last_block

        def forward(self, x : torch.Tensor):
            if x.size(-3) == 1:
                if len(x.size()) == 5: # if multiple input images
                    x = x.repeat(1, 1, 3, 1, 1)
                else:
                    x = x.repeat(1, 3, 1, 1)

            x = self.resnet_encoder(x)
            
            mu, log_var = x.view(x.size(0), -1).chunk(2, dim=-1)
            mu, log_var = mu.contiguous(), log_var.contiguous()
            
            return mu, log_var


    def __init__(self, **kwargs):
        if 'inference_without_encoder' in kwargs:
            kwargs.pop('inference_without_encoder')
        super(ResEnc_CBNN, self).__init__(inference_without_encoder=False, **kwargs)

    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            num_classes: int = 10,
            latent_dim: int = 256,
            inference_num_layers: int = 1,
            decoder_hidden_dims: List = None,
            batch_norm_track_running_stats: bool = True,
            **kwargs):

        resnet = ResNet18(in_channels=in_channels, num_classes=num_classes, batch_norm_track_running_stats=batch_norm_track_running_stats, **kwargs).resnet
        self.latent_dim = latent_dim
        self.inference_num_layers = inference_num_layers
        
        # Build modules
        self.context_encoder = ResEnc_CBNN.ResNetEncoder(resnet, latent_dim)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, decoder_hidden_dims)
        self.inference_encoder = self.context_encoder
        self.inference_classifier = BayesianClassifier(latent_dim*self.nb_input_images, num_classes, latent_dim, inference_num_layers, batch_norm_track_running_stats=batch_norm_track_running_stats)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(ResEnc_CBNN, cls).add_model_specific_args(parent_parser)

        parser = parent_parser.add_argument_group("ResEnc_CBNN")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--image_dim', type=int, default=64, help='Dimension of the input image. Image is assumed to be square.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space.')
        parser.add_argument('--inference_num_layers', type=int, default=1, help='Number of layers for the inference classifier.')
        parser.add_argument('--decoder_hidden_dims', type=int, nargs='+', default=[32, 64, 128, 256, 512], help='Hidden dimensions for the decoder.')
        parser.add_argument('--batch_norm_track_running_stats', type=bool, default=True, help='Track running statistics of batch normalisation layers.')
        
        return parent_parser


class MCQA_ResEnc_CBNN(ResEnc_CBNN):
    def __init__(self, **kwargs):
        super(MCQA_ResEnc_CBNN, self).__init__(**kwargs)

    def _init_modules(self,
            in_channels: int = 3,
            image_dim: int = 64,
            num_classes: int = 10,
            latent_dim: int = 256,
            inference_num_layers: int = 1,
            decoder_hidden_dims: List = None,
            batch_norm_track_running_stats: bool = True,
            **kwargs):

        resnet = ResNet18(in_channels=in_channels, num_classes=num_classes, batch_norm_track_running_stats=batch_norm_track_running_stats, **kwargs).resnet
        self.latent_dim = latent_dim
        self.inference_num_layers = inference_num_layers
        
        # Build modules
        self.context_encoder = ResEnc_CBNN.ResNetEncoder(resnet, latent_dim)
        self.context_decoder = CNNVariationalDecoder(self.recons_latent_dim, in_channels, image_dim, decoder_hidden_dims)
        self.inference_encoder = self.context_encoder
        self.inference_classifier = MCQABayesClassifier(latent_dim*self.nb_input_images, latent_dim, inference_num_layers, nb_context=self.nb_input_images-num_classes, nb_choices=num_classes, batch_norm_track_running_stats=batch_norm_track_running_stats)
    
