import torch.nn as nn
   
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []
        # Initial layer, no normalization
        layers.append(nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Downsampling layers
        for i in range(1, len(features)):
            layers.append(nn.Conv2d(features[i-1], features[i], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(features[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        # Final convolution to 1 output channel (PatchGAN)
        layers.append(nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)