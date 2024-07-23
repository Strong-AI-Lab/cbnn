
import math
from typing import Optional, Callable, Union, Type, List

import torch
from torch.nn.modules.utils import _pair, _triple
from torch.nn import Conv2d, Conv3d
from torch.nn.common_types import _size_2_t, _size_3_t
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
    

class BayesConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.log_var = torch.nn.Parameter(torch.randn_like(self.weight))
        self.bias_log_var = torch.nn.Parameter(torch.randn_like(self.bias)) if bias else None
        
        self.weight.data.normal_(0, 0.001 * 1/math.sqrt(in_channels))
        self.log_var.data.normal_(0, 0.001 * 1/math.sqrt(in_channels))
        if bias:
            self.bias.data.normal_(0, 0.001 * 1/math.sqrt(in_channels))
            self.bias_log_var.data.normal_(0, 0.001 * 1/math.sqrt(in_channels))
        
    
    def _sample_weights(self, mean: torch.Tensor, log_var: torch.Tensor, eps: Optional[torch.Tensor] = None):
        std = torch.exp(0.5 * log_var)

        if eps is None:
            eps = torch.randn_like(std)
        else:
            assert eps.shape == std.shape, f"Shape mismatch between eps and std: {eps.shape} != {std.shape}"

        return mean + eps * std
    
    def get_weight_distributions(self):
        if self.bias:
            w = torch.cat([self.weight.view(-1), self.bias.view(-1)])
            s = torch.cat([self.log_var.view(-1), self.bias_log_var.view(-1)])
        else:
            w = self.weight.view(-1)
            s = self.log_var.view(-1)
        return w, s
    
    def forward(self, x: torch.Tensor, eps: Optional[torch.Tensor] = None):
        weights = self._sample_weights(self.weight, self.log_var, eps)
        if self.bias is not None:
            bias = self._sample_weights(self.bias, self.bias_log_var, eps)
        else:
            bias = None
        
        return [self._conv_forward(x, weights, bias), weights] + ([bias] if bias is not None else [])


def bayesconv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> BayesConv2d:
    """3x3 Bayes convolution with padding"""
    return BayesConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BayesBlock(torch.nn.Module):
    """
    Basic block for Bayesian ResNet
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BayesBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BayesBlock")
        
        self.conv1 = bayesconv3x3(inplanes, planes, stride) # replace Conv2d with BayesConv2d
        self.bn1 = norm_layer(planes)
        self.silu = torch.nn.SiLU(inplace=True) # replace ReLU with SiLU
        self.conv2 = bayesconv3x3(planes, planes) # replace Conv2d with BayesConv2d
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def get_weight_distributions(self):
        m1, s1 = self.conv1.get_weight_distributions()
        m2, s2 = self.conv2.get_weight_distributions()
        return torch.cat([m1, m2]), torch.cat([s1, s2])

    def forward(self, x: torch.Tensor, eps: Optional[torch.Tensor] = None):
        identity = x

        out, *w1 = self.conv1(x, eps)
        out = self.bn1(out)
        out = self.silu(out)

        out, *w2 = self.conv2(out, eps)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)

        return [out, *w1, *w2]
    


class BayesResNet(ResNet):
    class BayesSequential(torch.nn.Sequential):
        def get_weight_distributions(self):
            m, s = [], []
            for module in self:
                mi, si = module.get_weight_distributions()
                m.append(mi)
                s.append(si)
            return torch.cat(m), torch.cat(s)

        def forward(self, input, eps: Optional[torch.Tensor] = None):
            weights = []
            for module in self:
                input, *w = module(input, eps)
                weights.extend(w)
            return [input, *w]

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
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.inplanes = 64
        self.conv1 = BayesConv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1 = self._make_bayes_layer(64, layers[0])
        self.inplanes = 256
        self.layer4 = self._make_bayes_layer(512, layers[3], stride=2, dilate=False if replace_stride_with_dilation is None else replace_stride_with_dilation[2])   

    def _make_bayes_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> BayesSequential:
        sequence = super()._make_layer(BayesBlock, planes, blocks, stride, dilate)
        return BayesResNet.BayesSequential(*sequence)

    def get_weight_distributions(self):
        m1, s1 = self.conv1.get_weight_distributions()
        m2, s2 = self.layer1.get_weight_distributions()
        m3, s3 = self.layer4.get_weight_distributions()
        return torch.cat([m1, m2, m3]), torch.cat([s1, s2, s3])

    def _forward_impl(self, x: torch.Tensor, eps: Optional[torch.Tensor] = None):
        x, *wc1 = self.conv1(x, eps)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, *wl1 = self.layer1(x, eps)
        x = self.layer2(x)
        x = self.layer3(x)
        x, *wl4 = self.layer4(x, eps)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        w = [*wc1, *wl1, *wl4]

        return x, w

    def forward(self, x: torch.Tensor, eps: Optional[torch.Tensor] = None):
        return self._forward_impl(x, eps)
    

class BayesResNetInvariant(BayesResNet):
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
        super().__init__(block, layers, in_channels, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.invariant_size = image_size // 4
        self.adapt_invariant = torch.nn.Linear(invariant_dim, 64 * self.invariant_size**2)
        self.inplanes = 128
        self.layer1 = self._make_bayes_layer(64, layers[0])

    def _forward_impl(self, x: torch.Tensor, r: torch.Tensor, eps: Optional[torch.Tensor] = None):
        batch_size = x.size(0)

        x, *wc1 = self.conv1(x, eps)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # concat with invariant representation
        r = self.adapt_invariant(r)
        r = r.view(batch_size, 64, self.invariant_size, self.invariant_size) # [B, C, H, W]
        x = torch.cat([x, r], dim=1)

        x, *wl1 = self.layer1(x, eps)
        x = self.layer2(x)
        x = self.layer3(x)
        x, *wl4 = self.layer4(x, eps)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        w = [*wc1, *wl1, *wl4]

        return x, *w

    def forward(self, x: torch.Tensor, r: torch.Tensor, eps: Optional[torch.Tensor] = None):
        return self._forward_impl(x, r, eps)
    




class MCQABayesResNetInvariant(BayesResNetInvariant):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], nb_input_images : int = 5, embedding_dim : int = 512, num_classes : int = 1000, **kwargs):
        super(MCQABayesResNetInvariant, self).__init__(block, layers, num_classes=embedding_dim, **kwargs)
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
    
    def forward(self, x : torch.Tensor, r: torch.Tensor, eps: Optional[torch.Tensor] = None): # [batch_size, nb_images, channels, height, width]
        batch_size = x.size(0)

        # Compute embeddings of context and choices
        x = x.view(batch_size * self.nb_input_images, *x.size()[2:]) # [batch_size, (nb_context + nb_choices), channels, height, width] -> [batch_size * (nb_context + nb_choices), channels, height, width]
        r = r.view(batch_size * self.nb_input_images, self.embedding_dim) # [batch_size, (nb_context + nb_choices), * invariant_dim] -> [batch_size * (nb_context + nb_choices), invariant_dim]
        x, *w = super().forward(x, r, eps) # [batch_size * (nb_context + nb_choices), channels, height, width] -> [batch_size * (nb_context + nb_choices), embedding_dim]
        x = x.view(batch_size, self.nb_input_images, self.embedding_dim) # [batch_size, (nb_context + nb_choices) * embedding_dim] -> [batch_size, nb_context + nb_choices, embedding_dim]

        context = x[:, :self.nb_context,:] # [batch_size, nb_context, embedding_dim]
        choices = x[:, self.nb_context:,:] # [batch_size, nb_choices, embedding_dim]
        
        if self.nb_context > 1:
            context = torch.nn.functional.silu(self.context_merger(context.view(batch_size, -1)))

        # Normalise embeddings
        context = context + torch.randn_like(context) * 1e-6 # Add eps*1e-6 to avoid division by zero
        choices = choices + torch.randn_like(choices) * 1e-6
        context = context / context.norm(dim=-1, keepdim=True)
        choices = choices / choices.norm(dim=-1, keepdim=True)

        # Compute scores
        scores = (context.unsqueeze(1) @ choices.permute(0,2,1)).view(batch_size, self.nb_choices) # [batch_size, 1, embedding_dim] x [batch_size, embedding_dim, nb_choices] -> [batch_size, nb_choices]
        return scores, *w
    




class MIPredBayesResNetInvariant(BayesResNetInvariant):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], nb_input_images : int = 5, embedding_dim : int = 512, num_classes : int = 1000, **kwargs):
        super(MIPredBayesResNetInvariant, self).__init__(block, layers, num_classes=embedding_dim, **kwargs)
        self.nb_input_images = nb_input_images
        self.embedding_dim = embedding_dim

        if self.nb_input_images < 2:
            raise ValueError("The number of input images must be at least 2.")

        self.fc_2 = torch.nn.Linear(nb_input_images * embedding_dim, num_classes)
    
    def forward(self, x : torch.Tensor, r: torch.Tensor, eps: Optional[torch.Tensor] = None): # [batch_size, nb_images, channels, height, width]
        batch_size = x.size(0)

        # Compute embeddings of context and choices
        x = x.view(batch_size * self.nb_input_images, *x.size()[2:]) # [batch_size, nb_input_images, channels, height, width] -> [batch_size * nb_input_images, channels, height, width]
        r = r.view(batch_size * self.nb_input_images, self.embedding_dim) # [batch_size, nb_input_images, * invariant_dim] -> [batch_size * nb_input_images, invariant_dim]
        x, *w = super().forward(x, r, eps) # [batch_size * nb_input_images, channels, height, width] -> [batch_size * nb_input_images, embedding_dim]
        x = x.view(batch_size, self.nb_input_images * self.embedding_dim) # [batch_size * nb_input_images, embedding_dim] -> [batch_size, nb_input_images * embedding_dim]
        
        # Compute classes
        classes = self.fc_2(x) # [batch_size, nb_input_images * embedding_dim] -> [batch_size, num_classes]
        return classes, *w


    

def bayes_resnet18(in_channels : int = 3, num_classes: int = 1000, **kwargs):
    return BayesResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, **kwargs)

def bayes_resnet18_invariant(in_channels : int = 3, num_classes: int = 1000, invariant_dim: int = 64, image_size: int = 64, **kwargs):
    return BayesResNetInvariant(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, invariant_dim=invariant_dim, image_size=image_size, **kwargs)

def mcqa_bayes_resnet18_invariant(in_channels : int = 3, nb_input_images: int = 5, embedding_dim: int = 512, num_classes: int = 1000, invariant_dim: int = 64, image_size: int = 64, **kwargs):
    return MCQABayesResNetInvariant(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, nb_input_images=nb_input_images, embedding_dim=embedding_dim, num_classes=num_classes, invariant_dim=invariant_dim, image_size=image_size, **kwargs)

def mipred_bayes_resnet18_invariant(in_channels : int = 3, nb_input_images: int = 5, embedding_dim: int = 512, num_classes: int = 1000, invariant_dim: int = 64, image_size: int = 64, **kwargs):
    return MIPredBayesResNetInvariant(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, nb_input_images=nb_input_images, embedding_dim=embedding_dim, num_classes=num_classes, invariant_dim=invariant_dim, image_size=image_size, **kwargs)