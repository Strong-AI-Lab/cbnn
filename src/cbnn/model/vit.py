
from typing import Optional, List

import pytorch_lightning as pl
import torchvision.models as models
import torch



VIT_MODELS = {
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,
    'vit_l_32': models.vit_l_32,
    'vit_h_14': models.vit_h_14,
}


class VIT(pl.LightningModule):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, version : str = 'vit_b_16', learning_rate: float = 0.005, weight_decay: float = 0.0, image_size : int = 224, patch_size : int = 16, freeze_parameters : Optional[List[str]] = None, reverse_freeze : bool = False, **kwargs):
        super(VIT, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.version = version
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.vit = VIT_MODELS[version](weights='DEFAULT' if self.pretrained else None)

        # Hacks to make it work with different input channels, image size and number of classes
        if num_classes != 1000:
            self.vit.heads.head = torch.nn.Linear(self.vit.heads.head.in_features, num_classes)
            print(f"Changed the output layer to match the number of classes: {num_classes}. Original number of classes: 1000.{' Pretrained weights overriden for this layer.' if pretrained else ''}") 

        self.modified_input = False
        if in_channels > 3 or in_channels == 2 or image_size != 224 or patch_size != 16:
            self.modified_input = True

            # Update input layer
            self.vit.conv_proj = torch.nn.Conv2d(in_channels=in_channels, out_channels=self.vit.hidden_dim, kernel_size=patch_size, stride=patch_size, padding=self.vit.conv_proj.padding)
            print(f"Changed the input layer to match the number of channels: {in_channels} (original number of channels: 3) or the image size: {image_size} (original image size: 224) or the patch size: {patch_size} (original patch size: 16).{' Pretrained weights overriden for this layer.' if pretrained else ''}")
            
            # Update attributes
            self.vit.image_size = image_size
            self.vit.patch_size = patch_size
            self.vit.seq_length = (image_size // patch_size)**2 + 1

            # Update positional embeddings
            self.vit.encoder.pos_embedding = torch.nn.Parameter(torch.empty(1, self.vit.seq_length, self.vit.hidden_dim).normal_(std=0.02))

        # Freeze parameters
        if freeze_parameters is not None:
            self.frozen_parameters = self.partial_freeze(freeze_parameters, reverse=reverse_freeze)
        else:
            self.frozen_parameters = None

        self.save_hyperparameters()

    def forward(self, x):
        # If the input has 1 channel, repeat the channels to match the input of the model
        if x.shape[1] == 1 and not self.modified_input:
            x = x.repeat(1, 3, 1, 1)

        # Process the input
        return self.vit(x)

    def loss_function(self, logits, y):
        return torch.nn.functional.cross_entropy(logits, y)

    def accuracy(self, logits, y):
        return torch.sum(torch.argmax(logits, dim=1) == y).item() / len(y)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("VIT")
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
        if self.frozen_parameters is not None:
            parameters = [param for name, param in self.named_parameters() if name not in self.frozen_parameters]
        else:
            parameters = self.parameters()
        return torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

    def partial_freeze(self, frozen_layers : List[str], reverse : bool = False):
        frozen = []
        for name, param in self.named_parameters():
            if (name in frozen_layers and not reverse) or (name not in frozen_layers and reverse):
                param.requires_grad = False
                frozen.append(name)
        return frozen
    


class MIPredVIT(VIT):
    def __init__(self, nb_input_images : int = 5, **kwargs):
        super(MIPredVIT, self).__init__(**kwargs)
        self.nb_input_images = nb_input_images

        if self.nb_input_images < 2:
            raise ValueError("The number of input images must be at least 2.")
        
        self.vit.seq_length = (self.vit.seq_length - 1) * self.nb_input_images + 1 # Change the sequence length to match the number of input images (add one for class token)
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
        if x.shape[1] == 1 and not self.modified_input:
            x = x.repeat(1, 1, 3, 1, 1)

        # Process the input
        return self.vit(x)

    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(MIPredVIT, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MIPredVIT")
        parser.add_argument('--nb_input_images', type=int, default=5, help='Number of input images.')
        return parent_parser





def with_predefined_version(version):
    def decorator(cls):
        class Subclass(cls):
            def __init__(self, *args, **kwargs):
                # Initialize with the predefined version
                super().__init__(*args, version=version, **kwargs)
        return Subclass
    return decorator


@with_predefined_version('vit_b_16')
class VITB16(VIT):
    pass

@with_predefined_version('vit_b_32')
class VITB32(VIT):
    pass

@with_predefined_version('vit_l_16')
class VITL16(VIT):
    pass

@with_predefined_version('vit_l_32')
class VITL32(VIT):
    pass

@with_predefined_version('vit_h_14')
class VITH14(VIT):
    pass

@with_predefined_version('vit_b_16')
class MIPredVITB16(MIPredVIT):
    pass

@with_predefined_version('vit_b_32')
class MIPredVITB32(MIPredVIT):
    pass

@with_predefined_version('vit_l_16')
class MIPredVITL16(MIPredVIT):
    pass

@with_predefined_version('vit_l_32')
class MIPredVITL32(MIPredVIT):
    pass

@with_predefined_version('vit_h_14')
class MIPredVITH14(MIPredVIT):
    pass