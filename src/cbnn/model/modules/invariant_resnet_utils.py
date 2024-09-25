
from typing import Type, Union, List, Optional, Callable

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet



class ResNetInvariant(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        invariant_dim: int = 64,
        image_size: int = 32,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.invariant_dim = invariant_dim
        self.image_size = image_size
        self.adapt_invariant = torch.nn.Linear(invariant_dim, image_size**2)
        
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(in_channels+1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x: torch.Tensor, r: torch.Tensor):
        batch_size = x.size(0)

        # concat with invariant representation
        r0 = self.adapt_invariant(r)
        r0 = r0.view(batch_size, 1, self.image_size, self.image_size)
        x = torch.cat([x, r0], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, torch.zeros(1) # need to add dummy weight for the loss function
    
    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parent_parser = super(ResNetInvariant, cls).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("ResNetInvariant")
        parser.add_argument('--invariant_dim', type=int, default=64, help='Dimension of the invariant representation.')
        parser.add_argument('--image_size', type=int, default=32, help='Size of the input images.')
        return parent_parser
    



def resnet18_invariant(in_channels : int = 3, num_classes: int = 1000, invariant_dim: int = 64, image_size: int = 64, **kwargs):
    return ResNetInvariant(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, invariant_dim=invariant_dim, image_size=image_size, **kwargs)
