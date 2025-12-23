from collections.abc import Sequence

import torch


def make_anchors(
    features: Sequence[torch.Tensor],
    strides: Sequence[int],
    grid_cell_offset: float = 0.5,
):
    """根據 Head 特徵圖的大小及其與原圖的縮放比例產生錨點和對應的步幅。

    Args:
        features (Sequence[torch.Tensor]):
            Head 特徵圖的張量序列，每個張量預期的形狀為`(B, C, H, W)`。
        strides (Sequence[int]):
            每個錨點之間實際的步幅。如果特徵圖長寬是輸入圖片的`1/8`，則步幅為`8`。
        grid_cell_offset (float, optional):
            錨點在每個網格單元內的偏移量，預設為`0.5`表示取網格中心為錨點座標。

    Returns:
        out (tuple[torch.Tensor, torch.Tensor]):
            1. 包含所有錨點座標的張量，形狀為`(num_anchors, 2)`。
            2. 每個錨點對應的步幅張量，形狀為`(num_anchors, 1)`。
    """
    dtype = features[0].dtype
    device = features[0].device
    anchor_tensors, stride_tensors = [], []
    for feat, stride in zip(features, strides, strict=True):
        h, w = feat.shape[-2:]
        sx = torch.arange(w, dtype=dtype) + grid_cell_offset  # shifted x scale
        sy = torch.arange(h, dtype=dtype) + grid_cell_offset  # shifted y scale
        # Makes grids with shape (h, w) containing x and y coordinates respectively.
        gx, gy = torch.meshgrid(sx, sy, indexing="xy")
        # Stacks them as one grid with shape (h, w, 2).
        grid = torch.stack((gx, gy), dim=-1)
        anchor_tensors.append(grid.view(h * w, 2))
        stride_tensors.append(torch.full((h * w, 1), stride, dtype=dtype))
    return (
        torch.cat(anchor_tensors).to(device),
        torch.cat(stride_tensors).to(device),
    )


def ltrb2xyxy(
    ltrb: torch.Tensor, anchor_points: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """將錨點對偵測框左、上、右、下邊界的距離轉換為偵測框左上與右下座標`(x0, y0, x1, y1)`。

    Args:
        ltrb (torch.Tensor):
            表示錨點與偵測框左、上、右、下邊界距離的張量，形狀為`(..., 4)`。
        anchor_points (torch.Tensor):
            表示錨點座標的張量，形狀為`(..., 2)`。
        dim (int, optional):
            偵測框距離與錨點座標所在的維度。預設為`-1`表示最後一個維度。

    Returns:
        out (torch.Tensor):
            表示偵測框左上與右下座標的張量，形狀為`(..., 4)`。
    """
    lt, rb = ltrb.chunk(2, dim=dim)
    x0y0 = anchor_points - lt
    x1y1 = anchor_points + rb
    return torch.cat((x0y0, x1y1), dim=dim)


def xyxy2ltrb(
    xyxy: torch.Tensor, anchor_points: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """將偵測框左上與右下座標`(x0, y0, x1, y1)`轉換為錨點對偵測框左、上、右、下邊界的偏差值。

    注意，此函式不會對偏差值進行任何限制，使用者根據實際需求處理。

    Args:
        xyxy (torch.Tensor):
            表示偵測框左上與右下座標的張量，形狀為`(..., 4)`。
        anchor_points (torch.Tensor):
            表示錨點座標的張量，形狀為`(..., 2)`。
        dim (int, optional):
            偵測框距離與錨點座標所在的維度。預設為`-1`表示最後一個維度。

    Returns:
        out (torch.Tensor):
            表示錨點與偵測框左、上、右、下邊界的偏差值張量，形狀為`(..., 4)`。
    """
    x0y0, x1y1 = xyxy.chunk(2, dim=dim)
    lt = anchor_points - x0y0
    rb = x1y1 - anchor_points
    return torch.cat((lt, rb), dim=dim)


def boxes_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, cxcywh: bool = False, eps: float = 1e-7
) -> torch.Tensor:
    """計算兩組偵測框之間的 IoU。

    此函式支援各種形狀的偵測框，根據兩組偵測框之間的形狀會產生不同意義的結果。

    以下條列幾種常見輸入形狀的使用情景及輸出形狀：
    1. `(4,)` & `(4,)`：
        計算兩個偵測框的 IoU，輸出為`(1,)`。
    2. `(N, 4)` & `(4,)`：
        計算`N`個偵測框與單一偵測框之間的 IoU，輸出為`(N,)`。
    3. `(N, 4)` & `(N, 4)`：
        計算`N`個偵測框兩兩之間的 IoU，輸出為`(N,)`。
    4. `(N, 1, 4)` & `(1, M, 4)`：
        計算`N`個偵測框與`M`個偵測框之間的 IoU，輸出為`(N, M)`。

    Args:
        boxes1 (torch.Tensor):
            表示一或多個偵測框的張量，最後一個維度大小必需是 4。
        boxes2 (torch.Tensor):
            表示一或多個偵測框的張量，最後一個維度大小必需是 4。
        cxcywh (bool, optional):
            若`True`則以`(cx, cy, w, h)`格式處理偵測框，否則以`(x0, y0, x1, y1)`格式。
            預設為`False`。
        eps (float, optional):
            用於避免除以零的極小值。預設為`1e-7`。

    Returns:
        out (torch.Tensor):
            表示 IoU 值的張量。
    """
    if cxcywh:
        # fmt: off
        b1_cx, b1_cy, b1_w, b1_h = boxes1.chunk(4, dim=-1)
        b2_cx, b2_cy, b2_w, b2_h = boxes2.chunk(4, dim=-1)
        b1_w2, b1_h2 = b1_w / 2, b1_h / 2
        b2_w2, b2_h2 = b2_w / 2, b2_h / 2
        b1_x0, b1_y0, b1_x1, b1_y1 = b1_cx - b1_w2, b1_cy - b1_h2, b1_cx + b1_w2, b1_cy + b1_h2
        b2_x0, b2_y0, b2_x1, b2_y1 = b2_cx - b2_w2, b2_cy - b2_h2, b2_cx + b2_w2, b2_cy + b2_h2
        # fmt: on
    else:
        b1_x0, b1_y0, b1_x1, b1_y1 = boxes1.chunk(4, dim=-1)
        b2_x0, b2_y0, b2_x1, b2_y1 = boxes2.chunk(4, dim=-1)
        b1_w, b1_h = b1_x1 - b1_x0, b1_y1 - b1_y0 + eps
        b2_w, b2_h = b2_x1 - b2_x0, b2_y1 - b2_y0 + eps

    inter_w = (b1_x1.minimum(b2_x1) - b1_x0.maximum(b2_x0)).clamp_(min=0)
    inter_h = (b1_y1.minimum(b2_y1) - b1_y0.maximum(b2_y0)).clamp_(min=0)
    inter_area = inter_w * inter_h
    union_area = b1_w * b1_h + b2_w * b2_h - inter_area + eps
    iou = inter_area / union_area
    return iou


def boxes_ciou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, cxcywh: bool = False, eps: float = 1e-7
) -> torch.Tensor:
    """計算兩組偵測框之間的 [Complete IoU](https://arxiv.org/abs/1911.08287v1)，請參考`boxes_iou`函式的說明。"""
    if cxcywh:
        # fmt: off
        b1_cx, b1_cy, b1_w, b1_h = boxes1.chunk(4, dim=-1)
        b2_cx, b2_cy, b2_w, b2_h = boxes2.chunk(4, dim=-1)
        b1_w2, b1_h2 = b1_w / 2, b1_h / 2
        b2_w2, b2_h2 = b2_w / 2, b2_h / 2
        b1_x0, b1_y0, b1_x1, b1_y1 = b1_cx - b1_w2, b1_cy - b1_h2, b1_cx + b1_w2, b1_cy + b1_h2
        b2_x0, b2_y0, b2_x1, b2_y1 = b2_cx - b2_w2, b2_cy - b2_h2, b2_cx + b2_w2, b2_cy + b2_h2
        # fmt: on
    else:
        b1_x0, b1_y0, b1_x1, b1_y1 = boxes1.chunk(4, dim=-1)
        b2_x0, b2_y0, b2_x1, b2_y1 = boxes2.chunk(4, dim=-1)
        b1_w, b1_h = b1_x1 - b1_x0, b1_y1 - b1_y0 + eps
        b2_w, b2_h = b2_x1 - b2_x0, b2_y1 - b2_y0 + eps

    inter_w = (b1_x1.minimum(b2_x1) - b1_x0.maximum(b2_x0)).clamp_(min=0)
    inter_h = (b1_y1.minimum(b2_y1) - b1_y0.maximum(b2_y0)).clamp_(min=0)
    inter_area = inter_w * inter_h
    union_area = b1_w * b1_h + b2_w * b2_h - inter_area + eps
    iou = inter_area / union_area

    # Compute squared length of smallest enclosing box diagonal.
    c2 = (
        (b1_x1.maximum(b2_x1) - b1_x0.minimum(b2_x0)).pow(2)
        + (b1_y1.maximum(b2_y1) - b1_y0.minimum(b2_y0)).pow(2)
        + eps
    )

    # Compute squared distance between the centers of the two bounding boxes.
    rho2 = (
        (b2_x0 + b2_x1 - b1_x0 - b1_x1).pow(2) + (b2_y0 + b2_y1 - b1_y0 - b1_y1).pow(2)
    ) / 4

    # Compute the consistency of aspect ratio between the two bounding boxes.
    v = (4 / torch.pi**2) * ((b2_w / b2_h).atan() - (b1_w / b1_h).atan()).pow(2)

    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def cxcywh2xyxy(
    boxes: torch.Tensor, split: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """將偵測框從中心座標格式`(cx, cy, w, h)`轉換為左上與右下座標格式`(x0, y0, x1, y1)`。

    Args:
        boxes (torch.Tensor):
            表示偵測框的張量，形狀為`(..., 4)`。
        split (bool, optional):
            是否將輸入張量拆分為四個部份回傳。預設為`False`。

    Returns:
        out (torch.Tensor):
            轉換後的偵測框張量，形狀為`(..., 4)`。
    """
    cx, cy, w, h = boxes.chunk(4, dim=-1)
    w_, h_ = w / 2, h / 2
    x0, y0 = cx - w_, cy - h_
    x1, y1 = cx + w_, cy + h_
    if split:
        return x0, y0, x1, y1
    return torch.cat((x0, y0, x1, y1), dim=-1)
