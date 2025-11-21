import torch
from torch import nn

from liandan.archs.common import autopad


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
        self.norm = nn.BatchNorm2d(co) if bn else nn.Identity()
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


class DetectionHead(nn.Module):
    def __init__(self, chs: tuple[int, ...], num_classes: int, reg_max=16):
        super().__init__()
        self.ni = len(chs)
        self.nc = num_classes
        self.reg_max = reg_max
        box_hidden = max(chs[0] // 4, self.reg_max * 4, 16)
        cls_hidden = max(chs[0], min(self.nc, 100))
        self.box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, box_hidden, 3),
                Conv(box_hidden, box_hidden, 3),
                nn.Conv2d(box_hidden, self.reg_max * 4, 1),
            )
            for c in chs
        )
        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, cls_hidden, 3),
                Conv(cls_hidden, cls_hidden, 3),
                nn.Conv2d(cls_hidden, self.nc, 1),
            )
            for c in chs
        )

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        ys = []
        for i in range(self.ni):
            box = self.box_convs[i](xs[i])
            cls = self.cls_convs[i](xs[i])
            ys.append(torch.cat((box, cls), dim=1))
        return ys


class YOLOv8(nn.Module):
    def __init__(self, num_classes: int = 80, reg_max: int = 16):
        super().__init__()
        # fmt: off
        self.backbone = nn.ModuleList(
            (
                Conv(  3,  16, s=2),                 # 0 -> P1 start (size /  2)
                Conv( 16,  32, s=2),                 # 1 -> P2 start (size /  4)
                C2f(  32,  32, n=1, shortcut=True),  # 2
                Conv( 32,  64, s=2),                 # 3 -> P3 start (size /  8)
                C2f(  64,  64, n=2, shortcut=True),  # 4
                Conv( 64, 128, s=2),                 # 5 -> P4 start (size / 16)
                C2f( 128, 128, n=2, shortcut=True),  # 6
                Conv(128, 256, s=2),                 # 7 -> P5 start (size / 32)
                C2f( 256, 256, n=1, shortcut=False), # 8
                SPPF(256, 256),                      # 9
            )
        )
        # fmt: on
        self.u10 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c11 = C2f(256 + 128, 128, n=1, shortcut=False)
        self.u12 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c13 = C2f(128 + 64, 64, n=1, shortcut=False)
        self.c14 = Conv(64, 64, s=2)
        self.c15 = C2f(64 + 128, 128, n=1, shortcut=False)
        self.c16 = Conv(128, 128, s=2)
        self.c17 = C2f(128 + 256, 256, n=1, shortcut=False)
        self.head = DetectionHead((64, 128, 256), num_classes, reg_max)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        backbone_outputs = {}
        for i, b in enumerate(self.backbone):
            x = b(x)
            if i in (4, 6, 9):
                backbone_outputs[i] = x

        x = self.u10(backbone_outputs[9])
        x = torch.cat((x, backbone_outputs[6]), dim=1)
        y11 = self.c11(x)
        x = self.u12(y11)
        x = torch.cat((x, backbone_outputs[4]), dim=1)
        y13 = self.c13(x)
        x = self.c14(y13)
        x = torch.cat((x, y11), dim=1)
        y15 = self.c15(x)
        x = self.c16(y15)
        x = torch.cat((x, backbone_outputs[9]), dim=1)
        y17 = self.c17(x)
        return self.head([y13, y15, y17])


if __name__ == "__main__":
    x = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    model = YOLOv8()
    ys = model(x)
    for y in ys:
        print(y.shape)
