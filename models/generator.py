import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += identity
        return out

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, n_residual_blocks=9):
        super(Generator, self).__init__()
        # Initial Convolution Block
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(features*2, features*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(features*4),
            nn.ReLU(inplace=True),
        )
        # Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(features*4) for _ in range(n_residual_blocks)])
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(features*2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )
        # Final Convolution Layer
        self.final = nn.Conv2d(features, out_channels, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return torch.tanh(self.final(x)) # Using tanh activation