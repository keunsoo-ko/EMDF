from .Unet import ConvNeXtV2
from .decoder import Generator
import torch
import torch.nn as nn


class Inpaint(nn.Module):
    def __init__(self, input_size=256, patch_size=16, depth=15, heads=16):
        super().__init__()
        self.coarse = ConvNeXtV2(4, 3)
        self.fine = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=input_size, img_channels=3)

    def forward(self, img, mask, is_stable=False):
        with torch.no_grad():
            c_gen, _ = self.coarse(img * (1 - mask), mask)
            c_gen = c_gen * mask + img * (1 - mask)

            z = torch.randn(img.shape[0], 512).to(img.device)
        gen = self.fine(c_gen, 1 - mask, z, None, noise_mode='random', return_stg1=False)
        return c_gen, gen
