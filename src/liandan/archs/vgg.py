from torch import nn


class VGGBlock(nn.Sequential):
    def __init__(self, num_conv, in_channels, out_channels):
        super().__init__()
        for _ in range(num_conv):
            self.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.append(nn.MaxPool2d(kernel_size=2, stride=2))
