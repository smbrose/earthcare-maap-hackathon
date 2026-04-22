# models/unet.py
# ---------------------------------------------------------------------
# File overview
# Standard UNet implementation for downscaling experiments.
# ---------------------------------------------------------------------
import torch.nn as nn
import torch

def conv_block(in_ch, out_ch):
    """
    Build a 2-layer convolutional block.

    Inputs:
        in_ch: input channels.
        out_ch: output channels.

    Outputs:
        nn.Sequential block with Conv-ReLU-Conv-ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Encoder
        self.c1 = conv_block(in_channels, 256)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(256, 512)
        self.p2 = nn.MaxPool2d(2)
        # Bottleneck
        self.b = conv_block(512, 1024)
        # Decoder
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d2 = conv_block(1024, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d1 = conv_block(512, 256)
        # Final
        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        b = self.b(p2)
        u2 = self.up2(b)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.d2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.d1(u1)
        out = self.out_conv(d1)
        return out
