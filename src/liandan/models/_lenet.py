import torch.nn as nn


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
