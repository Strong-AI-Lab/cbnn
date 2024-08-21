
import torch
import torch.nn.functional as F


class SquarePad:
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