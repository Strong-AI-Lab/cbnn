
from typing import List

import torch
import torch.nn.functional as F


class SquarePad:
    """
    Pad the input tensor to be square by adding zeros to the smaller dimension.
    """

    def __init__(self, fill : int = 0, padding_mode : str = 'constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img : torch.Tensor) -> torch.Tensor:
        _, h, w = img.size()
        max_wh = max(h, w)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, hp, vp, vp)
        return F.pad(img, padding, mode=self.padding_mode, value=self.fill)


class ImageConcat:
    """
    Display multiple images side by side on a pre-specified grid.
    """
    
    def __init__(self, nb_rows : int, nb_cols : int):
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

    def __call__(self, imgs : List[torch.Tensor]) -> torch.Tensor:
        if len(imgs) > self.nb_rows * self.nb_cols:
            raise ValueError(f"Expected {self.nb_rows * self.nb_cols} images maximum, got {len(imgs)}")
        elif len(imgs) < self.nb_rows * self.nb_cols:
            imgs.extend([torch.zeros_like(imgs[0]) for _ in range(self.nb_rows * self.nb_cols - len(imgs))])

        return torch.cat([torch.cat(imgs[i*self.nb_cols:(i+1)*self.nb_cols], dim=2) for i in range(self.nb_rows)], dim=1)