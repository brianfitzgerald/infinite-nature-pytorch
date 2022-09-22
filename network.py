import torch
import numpy as np
import torch.nn as nn
import config


class Network(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        num_channels = 16
        num_conv_blocks = 6


        layers = []
        for i in range(num_conv_blocks):
            b1 = nn.Conv2d(num_channels, 3, 1, 1)
            b2 = torch.instance_norm()
            b3 = nn.LeakyReLU()
            layers.extend([b1, b2, b3])

        layers.append(nn.Linear(config.DIM_OF_STYLE_EMBEDDING, config.DIM_OF_STYLE_EMBEDDING))
        
        self.layers = nn.Sequential(*layers)

    def encoder(self, x, scope="spade_encoder"):
        """Encoder that outputs global N(mu, sig) parameters.

        Args:
            x: [B, H, W, 4] an RGBD image (usually the initial image) which is used to
            sample noise from a distirbution to feed into the refinement
            network. Range [0, 1].
            scope: (str) variable scope

        Returns:
            (mu, logvar) are [B, 256] tensors of parameters defining a normal
            distribution to sample from.
        """

        # TODO rewrite
        
        x = 2 * x - 1
        
        x = self.layers(x)

        return x