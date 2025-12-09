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
    """將錨點對偵測框左上右下距離轉換為偵測框左上與右下座標`(x0, y0, x1, y1)`。"""
    lt, rb = ltrb.chunk(2, dim=dim)
    x0y0 = anchor_points - lt
    x1y1 = anchor_points + rb
    return torch.cat((x0y0, x1y1), dim=dim)


def bbox_iou(
    bbox1: torch.Tensor,
    bbox2: torch.Tensor,
    xywh: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    計算兩組 Bounding boxes 之間的 IoU。

    此函式支援各種形狀的`bbox1`及`bbox2`，只要最後一個維度是 4 即可。
    例如：可以傳入形狀為`(4,)`、`(N, 4)`、`(B, N, 4)`或`(B, N, 1, 4)`的張量。
    如果`xywh=True`程式內部會拆分最後一個維度為中心點`(x, y, w, h)`，否則為
    `(x0, y0, x1, y1)`。

    Args:
        bbox1 (torch.Tensor):
            表示一或多個 Bounding boxes 的張量，最後一個維度是 4。
        bbox2 (torch.Tensor):
            表示一或多個 Bounding boxes 的張量，最後一個維度是 4。
        xywh (bool, optional):
            若是`True`則以中心點`(x, y, w, h)`格式表示 Bounding boxes ，否則以
            `(x0, y0, x1, y1)`格式表示。
        eps (float, optional):
            用於避免除以零的極小值。

    Returns:
        out (torch.Tensor):
            表示 IoU 值的張量。
    """
    if xywh:
        # Converts (x, y, w, h) to (x0, y0, x1, y1).
        (x1, y1, w1, h1) = bbox1.chunk(4, dim=-1)
        (x2, y2, w2, h2) = bbox2.chunk(4, dim=-1)
        w1_, h1_ = w1 / 2, h1 / 2
        w2_, h2_ = w2 / 2, h2 / 2
        b1_x0, b1_y0, b1_x1, b1_y1 = x1 - w1_, y1 - h1_, x1 + w1_, y1 + h1_
        b2_x0, b2_y0, b2_x1, b2_y1 = x2 - w2_, y2 - h2_, x2 + w2_, y2 + h2_
    else:
        b1_x0, b1_y0, b1_x1, b1_y1 = bbox1.chunk(4, dim=-1)
        b2_x0, b2_y0, b2_x1, b2_y1 = bbox2.chunk(4, dim=-1)
        w1, h1 = b1_x1 - b1_x0, b1_y1 - b1_y0 + eps
        w2, h2 = b2_x1 - b2_x0, b2_y1 - b2_y0 + eps

    inter_x0 = (b1_x1.minimum(b2_x1) - b1_x0.maximum(b2_x0)).clamp_(min=0)
    inter_y0 = (b1_y1.minimum(b2_y1) - b1_y0.maximum(b2_y0)).clamp_(min=0)
    inter_area = inter_x0 * inter_y0
    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    return inter_area / union_area
