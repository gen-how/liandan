from functools import partial

import torch
import torch.nn as nn

from liandan.archs.common import autopad

# A BatchNorm2d constructor with the same eps and momentum values as official YOLOv8.
BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.03)

# YOLOv8 model size parameters: (depth gain, channel gain, ratio)
YOLOV8_PARAMS: dict[str, tuple[float, float, float]] = {
    "n": (1 / 3, 1 / 4, 2.0),
    "s": (1 / 3, 1 / 2, 2.0),
    "m": (2 / 3, 3 / 4, 1.5),
    "l": (1.00, 1.00, 1.00),
    "x": (1.00, 1.25, 1.00),
}


class Conv(nn.Module):
    DEFAULT_ACT = nn.SiLU(inplace=True)

    def __init__(
        self,
        ci: int,
        co: int,
        k: int | tuple[int, int] = 3,
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] | None = None,
        d: int | tuple[int, int] = 1,
        g: int = 1,
        bn: bool = True,
        act: bool | nn.Module = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, autopad(k, p, d), d, g, bias=False)
        self.norm = BatchNorm2d(co) if bn else nn.Identity()
        if act is True:
            self.act = self.DEFAULT_ACT
        elif act is False:
            self.act = nn.Identity()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            raise ValueError(f"Unsupported activation function: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        ci: int,
        co: int,
        ks: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (3, 3),
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        hidden = int(co * e)
        self.conv1 = Conv(ci, hidden, ks[0], 1)
        self.conv2 = Conv(hidden, co, ks[1], 1, g=g)
        self.shortcut = shortcut and ci == co
        assert self.shortcut == shortcut, "input channels != output channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class C2f(nn.Module):
    def __init__(
        self,
        ci: int,
        co: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        hidden = int(co * e)
        self.conv1 = Conv(ci, 2 * hidden, 1)
        self.conv2 = Conv((2 + n) * hidden, co, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(hidden, hidden, ((3, 3), (3, 3)), shortcut, g, e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = list(self.conv1(x).chunk(2, dim=1))
        # 傳生成器到 extend 函式邏輯上與以下程式碼相同:
        # for b in self.bottlenecks:
        #     outputs.append(b(outputs[-1]))
        outputs.extend(b(outputs[-1]) for b in self.bottlenecks)
        return self.conv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    def __init__(self, ci: int, co: int, k: int = 5):
        super().__init__()
        hidden = ci // 2
        self.conv1 = Conv(ci, hidden, 1)
        self.conv2 = Conv(hidden * 4, co, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), dim=1))


class Backbone(nn.Module):
    def __init__(self, version: str, in_channels: int = 3):
        super().__init__()
        assert version in YOLOV8_PARAMS.keys(), f"Unknown YOLOv8 version: {version}"
        d, w, r = YOLOV8_PARAMS[version]
        # fmt: off
        self.layers = nn.ModuleList([
            Conv( in_channels,     int( 64 * w),     s=2),                           # 0 (P1 start (size /  2))  # noqa: E501
            Conv(int( 64 * w),     int(128 * w),     s=2),                           # 1 (P2 start (size /  4))  # noqa: E501
            C2f( int(128 * w),     int(128 * w),     n=int(3 * d), shortcut=True),   # 2                         # noqa: E501
            Conv(int(128 * w),     int(256 * w),     s=2),                           # 3 (P3 start (size /  8))  # noqa: E501
            C2f( int(256 * w),     int(256 * w),     n=int(6 * d), shortcut=True),   # 4 -> output 1             # noqa: E501
            Conv(int(256 * w),     int(512 * w),     s=2),                           # 5 (P4 start (size / 16))  # noqa: E501
            C2f( int(512 * w),     int(512 * w),     n=int(6 * d), shortcut=True),   # 6 -> output 2             # noqa: E501
            Conv(int(512 * w),     int(512 * w * r), s=2),                           # 7 (P5 start (size / 32))  # noqa: E501
            C2f( int(512 * w * r), int(512 * w * r), n=int(3 * d), shortcut=False),  # 8                         # noqa: E501
            SPPF(int(512 * w * r), int(512 * w * r)),                                # 9 -> output 3             # noqa: E501
        ])
        # fmt: on
        self.output_indices = {4, 6, 9}

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.output_indices:
                outputs.append(x)
        return outputs


class Neck(nn.Module):
    def __init__(self, version: str):
        super().__init__()
        assert version in YOLOV8_PARAMS.keys(), f"Unknown YOLOv8 version: {version}"
        d, w, r = YOLOV8_PARAMS[version]
        # NOTE: All trainable layers disable shortcut in the Neck.
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # fmt: off
        self.c12 = C2f( int(512 * w * (1+r)), int(512 * w),     n=int(3 * d))
        self.c15 = C2f( int(768 * w),         int(256 * w),     n=int(3 * d))
        self.c16 = Conv(int(256 * w),         int(256 * w),     s=2)
        self.c18 = C2f( int(768 * w),         int(512 * w),     n=int(3 * d))
        self.c19 = Conv(int(512 * w),         int(512 * w),     s=2)
        self.c21 = C2f( int(512 * w * (1+r)), int(512 * w * r), n=int(3 * d))
        # fmt: on

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        # Output tensor from Backbone.
        b4, b6, b9 = xs
        # Treats upsample and torch.cat both as layers to calculate the order.
        x = self.upsample(b9)
        x = torch.cat((x, b6), dim=1)
        x = self.c12(x)
        x12 = x
        x = self.upsample(x)
        x = torch.cat((x, b4), dim=1)
        x = self.c15(x)
        y15 = x
        x = self.c16(x)
        x = torch.cat((x, x12), dim=1)
        x = self.c18(x)
        y18 = x
        x = self.c19(x)
        x = torch.cat((x, b9), dim=1)
        x = self.c21(x)
        y21 = x
        return [y15, y18, y21]


class Heads(nn.Module):
    def __init__(self, version: str, num_classes=80, reg_max=16):
        super().__init__()
        assert version in YOLOV8_PARAMS.keys(), f"Unknown YOLOv8 version: {version}"
        _, w, r = YOLOV8_PARAMS[version]
        self.nc = num_classes
        self.reg_max = reg_max
        chs = (int(256 * w), int(512 * w), int(512 * w * r))
        box_hidden = max(chs[0] // 4, self.reg_max * 4, 16)
        cls_hidden = max(chs[0], min(self.nc, 100))
        self.box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, box_hidden, k=3),
                Conv(box_hidden, box_hidden, k=3),
                nn.Conv2d(box_hidden, self.reg_max * 4, kernel_size=1),
            )
            for c in chs
        )
        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, cls_hidden, k=3),
                Conv(cls_hidden, cls_hidden, k=3),
                nn.Conv2d(cls_hidden, self.nc, kernel_size=1),
            )
            for c in chs
        )

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        for i in range(len(self.box_convs)):
            box = self.box_convs[i](xs[i])
            cls = self.cls_convs[i](xs[i])
            xs[i] = torch.cat((box, cls), dim=1)
        return xs


class YOLOv8(nn.Module):
    def __init__(
        self,
        version: str,
        num_classes: int = 80,
        img_size: tuple[int, int] = (640, 640),
        reg_max: int = 16,
    ):
        super().__init__()
        assert version in YOLOV8_PARAMS.keys(), f"Unknown YOLOv8 version: {version}"
        assert img_size[0] % 32 == 0, "Image width must be multiple of 32"
        assert img_size[1] % 32 == 0, "Image height must be multiple of 32"
        self.num_classes = num_classes
        self.img_size = img_size
        self.reg_max = reg_max
        self.strides = (8, 16, 32)
        self.backbone = Backbone(version)
        self.neck = Neck(version)
        self.heads = Heads(version, num_classes, reg_max)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        xs = self.heads(self.neck(self.backbone(x)))
        if self.training:
            return xs

        # TODO: Inference path.
        return []


if __name__ == "__main__":
    img_size = (640, 640)
    x = torch.zeros((1, 3, img_size[1], img_size[0]), dtype=torch.float32)
    model = YOLOv8(version="n", img_size=img_size)
    print(model)
    ys = model(x)
    for i, y in enumerate(ys):
        print(f"Head {i} shape = {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
