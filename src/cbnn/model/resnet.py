
from typing import List, Optional

import pytorch_lightning as pl
import torchvision.models as models
import torch

from .modules.norm import BatchNorm2dNoTrack


class ResNet18(pl.LightningModule):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, learning_rate : float = 0.005, weight_decay : float = 0.0, freeze_parameters : Optional[List[str]] = None, reverse_freeze : bool = False, batch_norm_track_running_stats : bool = True, **kwargs):
        super(ResNet18, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_norm_track_running_stats = batch_norm_track_running_stats

        norm_layer = None
        if batch_norm_track_running_stats == False:
            norm_layer = BatchNorm2dNoTrack

        self.resnet = models.resnet18(pretrained=pretrained, norm_layer=norm_layer)
        if num_classes != 1000:
            self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes) # Change the output layer to match the number of classes
        if in_channels > 3 or in_channels == 2:
            self.resnet.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Change the input layer to match the number of channels (no modifications needed if 3 channels; if 1 channel, repeat the channels during forward pass)

        # Freeze parameters
        if freeze_parameters is not None:
            self.partial_freeze(freeze_parameters, reverse=reverse_freeze)

        self.save_hyperparameters()

    def forward(self, x):
        # If the input has 1 channel, repeat the channels to match the input of the model
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Process the input
        return self.resnet(x)
    
    def loss_function(self, logits, y):
        return torch.nn.functional.cross_entropy(logits, y)
    
    def accuracy(self, logits, y):
        return torch.sum(torch.argmax(logits, dim=1) == y).item() / len(y)
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("ResNet18")
        parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels.')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset.')
        parser.add_argument('--pretrained', type=bool, default=True, help='Load pretrained weights.')
        parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the optimizer.')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the optimizer.')
        parser.add_argument('--freeze_parameters', type=str, nargs='+', default=None, help='List of layers to freeze.')
        parser.add_argument('--reverse_freeze', type=bool, default=False, help='Reverse the freezing of the layers.')
        parser.add_argument('--batch_norm_track_running_stats', type=bool, default=True, help='Track running statistics of batch normalisation layers.')
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



class MCQAResNet18(ResNet18):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, learning_rate : float = 0.005, weight_decay : float = 0.0, nb_input_images : int = 5, embedding_dim : int = 512, **kwargs):
        super(MCQAResNet18, self).__init__(in_channels, embedding_dim, pretrained, learning_rate, weight_decay, **kwargs)
        self.nb_input_images = nb_input_images
        self.nb_context = nb_input_images - num_classes
        self.nb_choices = num_classes
        self.embedding_dim = embedding_dim

        if self.nb_input_images < 2:
            raise ValueError("The number of input images must be at least 2.")
        
        if self.nb_choices < 2:
            raise ValueError("The number of choices must be at least 2.")
        
        if self.nb_context < 1:
            raise ValueError("The number of context images must be at least 1.")
        
        if self.nb_context > 1: # If there is more than one context image, we need to merge them
            self.context_merger = torch.nn.Linear(embedding_dim * self.nb_context, embedding_dim)
    
    def forward(self, x): # [batch_size, nb_images, channels, height, width]
        batch_size = x.size(0)

        # Compute embeddings of context and choices
        x = x.view(batch_size * self.nb_input_images, *x.size()[2:]) # [batch_size, (nb_context + nb_choices), channels, height, width] -> [batch_size * (nb_context + nb_choices), channels, height, width]
        x = super().forward(x) # [batch_size * (nb_context + nb_choices), channels, height, width] -> [batch_size * (nb_context + nb_choices), embedding_dim]
        x = x.view(batch_size, self.nb_input_images, self.embedding_dim) # [batch_size * (nb_context + nb_choices), embedding_dim] -> [batch_size, nb_context + nb_choices, embedding_dim]

        context = x[:, :self.nb_context,:] # [batch_size, nb_context, embedding_dim]
        choices = x[:, self.nb_context:,:] # [batch_size, nb_choices, embedding_dim]
        
        if self.nb_context > 1:
            context = torch.nn.functional.leaky_relu(self.context_merger(context.view(batch_size, -1)))

        # Normalise embeddings
        context = context + torch.randn_like(context) * 1e-6 # Add eps*1e-6 to avoid division by zero
        choices = choices + torch.randn_like(choices) * 1e-6
        context = context / context.norm(dim=-1, keepdim=True)
        choices = choices / choices.norm(dim=-1, keepdim=True)

        # Compute scores
        scores = (context.unsqueeze(1) @ choices.permute(0,2,1)).view(batch_size, self.nb_choices) # [batch_size, 1, embedding_dim] x [batch_size, embedding_dim, nb_choices] -> [batch_size, nb_choices]
        return scores

    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MCQAResNet18, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MCQAResNet18")
        parser.add_argument('--nb_input_images', type=int, default=5, help='Number of input images (context + choices).')
        parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the embeddings before weighting of the choices.')
        return parent_parser



class MIPredResNet18(ResNet18):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, learning_rate : float = 0.005, weight_decay : float = 0.0, nb_input_images : int = 5, embedding_dim : int = 512, **kwargs):
        super(MIPredResNet18, self).__init__(in_channels, embedding_dim, pretrained, learning_rate, weight_decay, **kwargs)
        self.nb_input_images = nb_input_images
        self.embedding_dim = embedding_dim

        if self.nb_input_images < 2:
            raise ValueError("The number of input images must be at least 2.")
        
        self.fc_2 = torch.nn.Linear(embedding_dim * self.nb_input_images, num_classes)
    
    def forward(self, x): # [batch_size, nb_images, channels, height, width]
        batch_size = x.size(0)

        # Compute embeddings of input images
        x = x.view(batch_size * self.nb_input_images, *x.size()[2:]) # [batch_size, nb_images, channels, height, width] -> [batch_size * nb_images, channels, height, width]
        x = super().forward(x) # [batch_size * nb_images, channels, height, width] -> [batch_size * nb_images, embedding_dim]
        x = x.view(batch_size, self.nb_input_images * self.embedding_dim) # [batch_size * nb_images, embedding_dim] -> [batch_size, nb_images * embedding_dim]

        # Compute classes
        classes = self.fc_2(x) # [batch_size, nb_images * embedding_dim] -> [batch_size, num_classes]
        return classes

    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MIPredResNet18, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MIPredResNet18")
        parser.add_argument('--nb_input_images', type=int, default=5, help='Number of input images (context + choices).')
        parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of the embeddings before weighting of the choices.')
        return parent_parser