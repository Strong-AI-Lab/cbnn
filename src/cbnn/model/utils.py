
import torch

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