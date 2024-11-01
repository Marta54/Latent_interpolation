import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 16, 8, 1),
            self._block(features_g * 16, features_g * 8, 16, 4, 1),
            self._block(features_g * 8, features_g * 4, 16, 4, 1),
            nn.ConvTranspose2d(features_g * 4, channels_img, kernel_size=8, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img,
                features_d,
                kernel_size = 4,
                stride = 2,
                padding = 1,
            ), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size = 4, stride = 2, padding = 1), # 16 x 16
            self._block(features_d * 2, features_d * 4, kernel_size = 4, stride = 2, padding = 1), # 8 x 8
            self._block(features_d * 4, features_d * 8, kernel_size = 4, stride = 2, padding = 1), # 4 x 4
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(features_d * 8, 1, kernel_size = 4, stride = 2, padding = 0), # conversts into single value that represents if image is fake or real
            nn.Sigmoid()
            )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), # slope of 0.2
        )
    
    def forward(self, x):
        return self.disc(x)
    
    
def initialize_weights(model):
    for m in model.modules(): 
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

import time
def test():
    N, in_channels, H, W = 8, 3, 552, 552
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
   
    gen = Generator(z_dim, in_channels, 16)
    z = torch.randn(N, z_dim, 1, 1)
    t0 = time.time()
    x = gen(z)
    print(time.time() - t0)
    print(x.shape)
    print(x.shape == (N, in_channels, H, W))
    print('success')

    disc  = Discriminator(in_channels, 8)
    initialize_weights(disc)
    print(disc(x).shape)
    print(disc(x).shape == (N,1,1,1))

#test()