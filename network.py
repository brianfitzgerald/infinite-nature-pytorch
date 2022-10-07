import torch
import numpy as np
import torch.nn as nn
import config
import spade


class Network(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        num_channels = 16
        num_conv_blocks = 6

        block_sizes = [1, 2, 4, 8, 8, 8]

        layers = []
        for i in range(num_conv_blocks):
            b1 = nn.Conv2d(block_sizes[i], 3, 1, 1)
            b2 = torch.instance_norm()
            b3 = nn.LeakyReLU(0.2)
            layers.extend([b1, b2, b3])

        self.layers = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(config.DIM_OF_STYLE_EMBEDDING)
        self.logvar = nn.Linear(config.DIM_OF_STYLE_EMBEDDING)

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

        x = self.layers(x)
        mu = self.mu_layer(x)
        logvar = self.logvar(x)
        return mu, logvar

    def refinement(self, rgbd, mask, z):
        """Refines rgbd, mask based on noise z.

        H, W should be divisible by 2 ** num_up_layers

        Args:
            rgbd: [B, H, W, 4] the rendered view to be refined
            mask: [B, H, W, 1] binary mask of unknown regions. 1 where known and 0 where
            unknown
            z: [B, D] a noise vector to be used as noise for the generator
            scope: (str) variable scope

        Returns:
            [B, H, W, 4] refined rgbd image.
        """

        img = 2 * rgbd - 1
        img = torch.concat([img, mask], axis=-1)

        num_channel = 32

        num_up_layers = 5
        out_channels = 4  # For RGBD

        batch_size, im_height, im_width, unused_c = rgbd.get_shape().as_list()

        init_h = im_height // (2 ** num_up_layers)
        init_w = im_width // (2 ** num_up_layers)

        x = torch.linear(z, 16 * init_h * init_w)
        x = torch.reshape(x, [batch_size, init_h, init_w, 16 * num_channel])
        spade_blocks = [16, 16, 16, 8, 4, 2, 1]
        for i in range(spade_blocks):
            x = spade.SPADEResnetBlock(x, img, spade_blocks[i] * num_channel)
        x = torch.nn.functional.leaky_relu(x, 0.2)
        x = torch.nn.functional.conv2d(x, out_channels, 3, 1, 1)
        x = torch.tanh(x)

        return 0.5 * (x + 1)