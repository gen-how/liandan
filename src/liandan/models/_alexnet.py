import torch.nn as nn


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
