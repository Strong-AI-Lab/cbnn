
import os

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