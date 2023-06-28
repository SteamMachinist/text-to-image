# Код генератора из GAN-CLS.ipynb

import torch
from torch import nn


class ConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation,
                 if_act=True,
                 if_batch_norm=True):
        super().__init__()
        layers = []
        conv_tr_2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=False)
        batch_norm = nn.BatchNorm2d(out_channels)
        layers.append(conv_tr_2d)
        if if_batch_norm:
            layers.append(batch_norm)
        if if_act:
            layers.append(activation)
        self.conv_transpose2d = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_transpose2d(x)


class LinearProjection(nn.Module):
    def __init__(self,
                 in_embedding_dim,
                 projected_embedding_dim,
                 activation):
        super().__init__()
        projection = nn.Linear(in_features=in_embedding_dim, out_features=projected_embedding_dim)
        batch_norm = nn.BatchNorm1d(projected_embedding_dim)
        self.linear_proj = nn.Sequential(projection, batch_norm, activation)

    def forward(self, x):
        return self.linear_proj(x)


class Generator(nn.Module):
    def __init__(self,
                 in_embedding_dim,
                 projected_embedding_dim,
                 noise_dim,
                 channels_dim,
                 out_channels):
        super(Generator, self).__init__()
        concat_dim = projected_embedding_dim + noise_dim
        self.linear_projection = LinearProjection(in_embedding_dim=in_embedding_dim,
                                                  projected_embedding_dim=projected_embedding_dim,
                                                  activation=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.generator = nn.Sequential(
            ConvTranspose2D(in_channels=concat_dim, out_channels=channels_dim * 8, kernel_size=4, stride=1, padding=0,
                            activation=nn.ReLU(inplace=True)),
            # Output: (channels_dim*8) x 4 x 4
            ConvTranspose2D(in_channels=channels_dim * 8, out_channels=channels_dim * 4, kernel_size=4, stride=2,
                            padding=1, activation=nn.ReLU(inplace=True)),  # Output: (channels_dim*4) x 8 x 8
            ConvTranspose2D(in_channels=channels_dim * 4, out_channels=channels_dim * 2, kernel_size=4, stride=2,
                            padding=1, activation=nn.ReLU(inplace=True)),
            # Output: (channels_dim*2) x 16 x 16
            ConvTranspose2D(in_channels=channels_dim * 2, out_channels=channels_dim, kernel_size=4, stride=2, padding=1,
                            activation=nn.ReLU(inplace=True)),
            # Output: (channels_dim) x 32 x 32
            ConvTranspose2D(in_channels=channels_dim, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                            activation=nn.Tanh(), if_batch_norm=False)  # Output: (out_channels) x 64 x 64
        )

    def forward(self, input_embedding, noise):
        x = self.linear_projection(input_embedding).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, noise], 1)
        x = self.generator(x)
        return x
