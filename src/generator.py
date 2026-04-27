import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, down=True):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, output_padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    def __init__(self, img_channels=3, num_residuals=9):
        super().__init__()

        # Initial Layer
        self.initial = ConvBlock(img_channels, 64, kernel_size=7, stride=1, padding=3)

        # Down Sampling
        in_channels = 64
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, in_channels*2, 3, 2, 1),
                ConvBlock(in_channels*2, in_channels*4, 3, 2, 1),
            ]
        )

        # Residual Sampling
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)] 
        )

        # Up Sampling
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(256, 128, 3, 2, 1, down=False),
                ConvBlock(128, 64, 3, 2, 1, down=False)
            ]
        ) 

        # Last Block
        self.last = nn.Conv2d(64, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)

        return torch.tanh(self.last(x))


