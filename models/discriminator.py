import torch
import torch.nn as nn

# Adding Discriminator for baseline model comparison
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = []

        # Initial Layer
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Hidden Layers
        in_features = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_features, feature, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_features = feature

        # Final Layer
        layers.append(
            nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)