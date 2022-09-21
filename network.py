import torch
import numpy as np


class Network(torch.nn.Module):

    def encoder(x, scope="spade_encoder"):
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
        num_channels = 16
        
        x = torch.nn.Conv2d(4, num_channels, 3, 1, 1)(x)
        x = torch.instance_norm(x)
        x = torch.nn.ReLU()(x)

        x = torch.nn.Conv2d(4, num_channels, 3, 1, 1)(x)
        x = torch.instance_norm(x)
        x = torch.nn.ReLU()(x)


        x = torch.nn.Conv2d(4, num_channels, 3, 1, 1)(x)
        x = torch.instance_norm(x)
        x = torch.nn.ReLU()(x)