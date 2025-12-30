import torch.nn as nn


class VGGBlock(nn.Sequential):
    def __init__(self, num_conv, in_channels, out_channels):
        super().__init__()
        for _ in range(num_conv):
            self.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.append(nn.MaxPool2d(kernel_size=2, stride=2))


class VGG16(nn.Sequential):
    def __init__(self):
        # fmt: off
        super().__init__(
            # 64 x 112 x 112
            VGGBlock(2, 3, 64),
            # 128 x 56 x 56
            VGGBlock(2, 64, 128),
            # 256 x 28 x 28
            VGGBlock(3, 128, 256),
            # 512 x 14 x 14
            VGGBlock(3, 256, 512),
            # 512 x 7 x 7
            VGGBlock(3, 512, 512), nn.Flatten(),
            # 4096
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            # 4096
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            # 1000
            nn.Linear(4096, 1000)
        )
        # fmt: on
