# models/unetskip.py
# ---------------------------------------------------------------------
# File overview
# UNet with skip connections used for downscaling experiments.
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


class UNetSkip(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        # Encoder
        self.enc1 = conv_block(in_channels, 256)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(256, 512)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # After concat: 512 (upsampled) + 512 (skip) = 1024
        self.dec2 = conv_block(1024, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # After concat: 256 (upsampled) + 256 (skip) = 512
        self.dec1 = conv_block(512, 256)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # [B,256,H,W]
        p1 = self.pool1(e1)        # [B,256,H/2,W/2]
        e2 = self.enc2(p1)         # [B,512,H/2,W/2]
        p2 = self.pool2(e2)        # [B,512,H/4,W/4]

        # Bottleneck
        b = self.bottleneck(p2)    # [B,1024,H/4,W/4]

        # Decoder with skip connections
        u2 = self.up2(b)           # [B,512,H/2,W/2]
        u2 = torch.cat([u2, e2], dim=1)  # [B,1024,H/2,W/2]
        d2 = self.dec2(u2)         # [B,512,H/2,W/2]

        u1 = self.up1(d2)          # [B,256,H,W]
        u1 = torch.cat([u1, e1], dim=1)  # [B,512,H,W]
        d1 = self.dec1(u1)         # [B,256,H,W]

        # Output
        return self.out_conv(d1)   # [B,out_channels,H,W]  
