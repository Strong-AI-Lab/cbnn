
from typing import Optional, List

import pytorch_lightning as pl
import torchvision.models as models
import torch


class VITB16(pl.LightningModule):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, learning_rate: float = 0.005, weight_decay: float = 0.0, image_size : int = 224, patch_size : int = 16, freeze_parameters : Optional[List[str]] = None, reverse_freeze : bool = False, **kwargs):
        super(VITB16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if patch_size == 16 and image_size == 224:
            self.vit = models.vit_b_16(pretrained=pretrained)

            if num_classes != 1000:
                self.vit.heads.head = torch.nn.Linear(self.vit.heads.head.in_features, num_classes, bias=True)  # Change the output layer to match the number of classes
        
        elif pretrained:
            raise ValueError("The pretrained weights are only available for the ViT-B/16 model with a patch size of 16 and an image size of 224.")
        else:
            self.vit = models.VisionTransformer(image_size=image_size, patch_size=patch_size, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=num_classes)


        if in_channels > 3 or in_channels == 2:
            self.vit.conv_proj = torch.nn.Conv2d(in_channels, self.vit.hidden_dim, self.vit.conv_proj.kernel_size, self.vit.conv_proj.stride, self.vit.conv_proj.padding, bias=False)  # Change the input layer to match the number of channels (no modifications needed if 3 channels; if 1 channel, repeat the channels during forward pass)

        # Freeze parameters
        if freeze_parameters is not None:
            self.partial_freeze(freeze_parameters, reverse=reverse_freeze)

        self.save_hyperparameters()

    def forward(self, x):
        # If the input has 1 channel, repeat the channels to match the input of the model
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Process the input
        return self.vit(x)

    def loss_function(self, logits, y):
        return torch.nn.functional.cross_entropy(logits, y)

    def accuracy(self, logits, y):
        return torch.sum(torch.argmax(logits, dim=1) == y).item() / len(y)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("VITB16")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--pretrained', type=bool, default=True, help='Load pretrained weights.')
        parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer.')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
        parser.add_argument('--image_size', type=int, default=224, help='Size of the input images.')
        parser.add_argument('--patch_size', type=int, default=16, help='Size of the patches.')
        parser.add_argument('--freeze_parameters', type=str, nargs='+', default=None, help='List of layers to freeze.')
        parser.add_argument('--reverse_freeze', type=bool, default=False, help='Reverse the freezing of the layers.')
        return parent_parser
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        acc = self.accuracy(logits, y)
        self.log('train_loss', loss)
        self.log('train_Accuracy', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_Accuracy', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_function(logits, y)
        acc = self.accuracy(logits, y)
        self.log('test_loss', loss)
        self.log('test_Accuracy', acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def partial_freeze(self, frozen_layers : List[str], reverse : bool = False):
        for name, param in self.named_parameters():
            if (name in frozen_layers and not reverse) or (name not in frozen_layers and reverse):
                param.requires_grad_(False)
    


class MIPredVITB16(VITB16):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, learning_rate : float = 0.005, weight_decay : float = 0.0, nb_input_images : int = 5, **kwargs):
        super(MIPredVITB16, self).__init__(in_channels, num_classes, pretrained, learning_rate, weight_decay, **kwargs)
        self.nb_input_images = nb_input_images
        self.nb_context = nb_input_images - num_classes
        self.nb_choices = num_classes

        if self.nb_input_images < 2:
            raise ValueError("The number of input images must be at least 2.")
        
        if self.nb_choices < 2:
            raise ValueError("The number of choices must be at least 2.")
        
        if self.nb_context < 1:
            raise ValueError("The number of context images must be at least 1.")
        
        self.vit.seq_length = (self.vit.seq_length - 1 ) * self.nb_input_images + 1 # Change the sequence length to match the number of input images (add one for class token)
        self.vit.encoder.pos_embedding = torch.nn.Parameter(torch.empty(1, self.vit.seq_length, self.vit.hidden_dim).normal_(std=0.02))
        self.vit._process_input = self._process_input
            
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = x.shape # [batch_size, nb_images, channels, height, width]
        p = self.vit.patch_size
        torch._assert(h == self.vit.image_size, f"Wrong image height! Expected {self.vit.image_size} but got {h}!")
        torch._assert(w == self.vit.image_size, f"Wrong image width! Expected {self.vit.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # [b, n, c, h, w] -> [b, n, hidden_dim, n_h * n_w]
        x = x.reshape(b * n, c, h, w)
        x = self.vit.conv_proj(x)
        x = x.reshape(b, n, self.vit.hidden_dim, n_h * n_w)

        # [b, n, hidden_dim, (n_h * n_w)] -> [b , (n * n_h * n_w), hidden_dim]
        # The self attention layer expects inputs in the format [N, S, E]
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 1, 3, 2).reshape(b, n * n_h * n_w, self.vit.hidden_dim)

        return x

    def forward(self, x):
        # If the input has 1 channel, repeat the channels to match the input of the model
        if x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        # Process the input
        return self.vit(x)

    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MIPredVITB16, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MIPredVITB16")
        parser.add_argument('--nb_input_images', type=int, default=5, help='Number of input images (context + choices).')
        return parent_parser

