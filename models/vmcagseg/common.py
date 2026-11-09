import torch
import torch.nn as nn
import torch.nn.functional as F

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


def downsample(x, ratios=[4, 8, 16]):
    x_scales = [x]
    _, _, H, W = x.shape
    for i_layer in range(len(ratios)):
        ratio = ratios[i_layer]
        x_scale = F.interpolate(x, (H//ratio, W//ratio))
        x_scales.append(x_scale)
    return x_scales