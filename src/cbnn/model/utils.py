
import os
from typing import Callable, List

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


COLOR_CODE = [
            [0.0, 0.0, 0.0], 
            [0.75, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.75, 1.0, 0.0], 
            [0.25, 0.5, 0.5], 
            [1.0, 0.5, 0.5], 
            [0.0, 0.0, 1.0], 
            [0.75, 0.0, 1.0], 
            [0.0, 1.0, 1.0], 
            [0.75, 1.0, 1.0]] # each of the 10 classes has an attributed rgb code, adapted from http://hydra.nat.uni-magdeburg.de/packing/scu/scu10.html and normalised to [0, 1]

def tensor_to_rgb(x : torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of shape (B, C, H, W) to a tensor of shape (B, 3, H, W) with the rgb values of the color code.
    Assumes C > 3
    """
    nb_channels = x.shape[0]
    code = torch.tensor(COLOR_CODE, dtype=torch.float32, device=x.device)[:nb_channels]

    x = x.permute(0, 2, 3, 1)
    x = torch.matmul(x,code)
    return x.permute(0, 3, 1, 2)


def sample_images(model : torch.nn.Module, data : torch.Tensor, log_dir : str, log_name : str, epoch : int, num_samples: int = 64):
    try:
        device = next(model.parameters()).device
        test_input = data.to(device)

        # Format image if needed
        if test_input.shape[1] > 3:
            test_input_rgb = tensor_to_rgb(test_input)
        else:
            test_input_rgb = test_input

        # Save input images
        os.makedirs(os.path.join(log_dir, "Input_Images"), exist_ok=True)
        vutils.save_image(test_input_rgb.data,
                        os.path.join(log_dir, 
                                    "Input_Images", 
                                    f"input_{log_name}_Epoch_{epoch}.png"),
                        normalize=True,
                        nrow=12)

        # Generate reconstruction images
        recons = model.generate(test_input)
        if recons.shape[1] > 3:
            recons = tensor_to_rgb(recons)

        os.makedirs(os.path.join(log_dir, "Reconstructions"), exist_ok=True)
        vutils.save_image(recons.data,
                        os.path.join(log_dir, 
                                    "Reconstructions", 
                                    f"recons_{log_name}_Epoch_{epoch}.png"),
                        normalize=True,
                        nrow=12)
    
        # Generate samples
        latent_dim = model.recons_latent_dim if hasattr(model, 'recons_latent_dim') else model.latent_dim
        samples = model.decode(torch.randn(num_samples,latent_dim).to(device))
        if samples.shape[1] > 3:
            samples = tensor_to_rgb(samples)

        os.makedirs(os.path.join(log_dir, "Samples"), exist_ok=True)
        vutils.save_image(samples.cpu().data,
                        os.path.join(log_dir, 
                                    "Samples",      
                                    f"{log_name}_Epoch_{epoch}.png"),
                        normalize=True,
                        nrow=12)
    
    except StopIteration:
        pass





def average_collage_optim(input_collator : Callable): # to use carefully, can lead to poor performance if classifier uses batch normalisation
    def collate_fn(zs : List[torch.Tensor], z_cs : List[torch.Tensor], classifier : Callable):
        context_size = len(zs)
        batch_size = zs[0].shape[0]
        z = torch.cat(zs, dim=0)
        z_c = torch.cat(z_cs, dim=0)
        inputs = input_collator(z, z_c, classifier)
        outputs, *w = classifier(*inputs)
        outputs = outputs.view(context_size, batch_size, -1).mean(dim=0)
        return outputs, *w

    return collate_fn

def average_collage(input_collator : Callable): # performs multiple weight samplings
    def collate_fn(zs : List[torch.Tensor], z_cs : List[torch.Tensor], classifier : Callable):
        outputs = []
        ws = []
        for i in range(len(zs)):
            inputs = input_collator(zs[i], z_cs[i], classifier)
            output, *w = classifier(*inputs)
            outputs.append(output)
            ws.extend(w)
        outputs = torch.stack(outputs).mean(dim=0)
        return outputs, *ws

    return collate_fn  


@average_collage
def cat_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [torch.cat([z, z_c], dim=-1)]

@average_collage
def sum_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z + z_c]

@average_collage_optim
def cat_collator_optim(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [torch.cat([z, z_c], dim=-1)]

@average_collage_optim
def sum_collator_optim(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z + z_c]

@average_collage
def mul_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z * z_c]

@average_collage
def sub_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z - z_c]

@average_collage
def none_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z, z_c]


def cross_product(zs : List[torch.Tensor]) -> torch.Tensor:
    if len(zs) != zs[0].size(-1) -1:
        raise ValueError(f"Cross collator uses the generalised cross product and requires to have n-1 vectors of size n, had {len(zs)} vectors of size {zs[0].size(-1)}")
    dim = zs[0].size(-1)
    
    context = torch.stack(zs, dim=1) # [B, n-1, n]

    # Normalise the context matrices
    context = context / torch.norm(context, dim=0, keepdim=True)

    # Compute the cross product of the context matrices
    submatrices = torch.stack([context[:,:,torch.arange(dim)!=i] for i in range(dim)]).permute(1, 0, 2, 3) # [B, n, n-1, n-1]
    dets = torch.linalg.det(submatrices) # [B, n]
    cross = dets * (-torch.pow(-1, torch.arange(dets.size(1),device=dets.device))).view(1,-1) # [B, n]

    return cross


def cross_collage(input_collator : Callable):
    def collate_fn(zs : List[torch.Tensor], z_cs : List[torch.Tensor], classifier : Callable):
        z_cs = cross_product(z_cs) # [B, n] * C -> [B, n]
        z_cs = z_cs.unsqueeze(1).repeat(1, len(zs), 1).unbind(dim=1) # [B, n] -> [B, n] * C
        return input_collator(zs, z_cs, classifier)
    
    return collate_fn


@cross_collage
@average_collage
def cross_cat_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [torch.cat([z, z_c], dim=-1)]

@cross_collage
@average_collage
def cross_sum_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z + z_c]

@cross_collage
@average_collage
def cross_mul_collator(z : torch.Tensor, z_c : torch.Tensor, classifier : Callable):
    return [z * z_c]



INFERENCE_CONTEXT_COLLATORS = {
    "cat" : cat_collator,
    "cat_optim" : cat_collator_optim,
    "sum" : sum_collator,
    "sum_optim" : sum_collator_optim,
    "mul" : mul_collator,
    "sub" : sub_collator,
    "none" : none_collator,
    "cross_cat" : cross_cat_collator,
    "cross_sum" : cross_sum_collator,
    "cross_mul" : cross_mul_collator,
}