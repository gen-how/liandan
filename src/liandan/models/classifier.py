from torch import nn

from ..archs.conv import VGGBlock


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


class AlexNet(nn.Sequential):
    def __init__(self):
        # fmt: off
        super().__init__(
            # 96 x 55 x 55
            nn.Conv2d(3, 48 * 2, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            # 96 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256 x 27 x 27
            nn.Conv2d(48 * 2, 128 * 2, kernel_size=5, padding=2), nn.ReLU(),
            # 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 384 x 13 x 13
            nn.Conv2d(128 * 2, 192 * 2, kernel_size=3, padding=1), nn.ReLU(),
            # 384 x 13 x 13
            nn.Conv2d(192 * 2, 192 * 2, kernel_size=3, padding=1), nn.ReLU(),
            # 256 x 13 x 13
            nn.Conv2d(192 * 2, 128 * 2, kernel_size=3, padding=1), nn.ReLU(),
            # 256 x 6 x 6
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            # 4096
            nn.Linear(256 * 6 * 6, 2048 * 2), nn.ReLU(), nn.Dropout(p=0.5),
            # 4096
            nn.Linear(2048 * 2, 2048 * 2), nn.ReLU(), nn.Dropout(p=0.5),
            # 1000
            nn.Linear(2048 * 2, 1000)
        )
        # fmt: on


class LeNet(nn.Sequential):
    def __init__(self):
        # fmt: off
        super().__init__(
            # C1: 6 x 28 x 28
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            # S2: 6 x 14 x 14
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3: 16 x 10 x 10
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            # S4: 16 x 5 x 5
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            # F5: 120
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            # F6: 84
            nn.Linear(120, 84), nn.Sigmoid(),
            # Out: 10
            nn.Linear(84, 10),
        )
        # fmt: on
