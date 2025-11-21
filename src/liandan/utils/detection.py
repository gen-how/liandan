import torch


def bbox_iou(
    box1: torch.Tensor, box2: torch.Tensor, xywh=False, eps=1e-7
) -> torch.Tensor:
    """
    計算兩組 Bounding boxes 之間的 IoU。

    此函式支援各種形狀的`box1`及`box2`，只要最後一個維度是 4 即可。
    例如：可以傳入形狀為`(4,)`、`(N, 4)`、`(B, N, 4)`或`(B, N, 1, 4)`的張量。
    如果`xywh=True`程式內部會拆分最後一個維度為中心點`(x, y, w, h)`，否則為
    `(x0, y0, x1, y1)`。

    Args:
        box1 (torch.Tensor):
            表示一或多個 Bounding boxes 的張量，最後一個維度是 4。
        box2 (torch.Tensor):
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
        (x1, y1, w1, h1) = box1.chunk(4, dim=-1)
        (x2, y2, w2, h2) = box2.chunk(4, dim=-1)
        w1_, h1_ = w1 / 2, h1 / 2
        w2_, h2_ = w2 / 2, h2 / 2
        b1_x0, b1_y0, b1_x1, b1_y1 = x1 - w1_, y1 - h1_, x1 + w1_, y1 + h1_
        b2_x0, b2_y0, b2_x1, b2_y1 = x2 - w2_, y2 - h2_, x2 + w2_, y2 + h2_
    else:
        b1_x0, b1_y0, b1_x1, b1_y1 = box1.chunk(4, dim=-1)
        b2_x0, b2_y0, b2_x1, b2_y1 = box2.chunk(4, dim=-1)
        w1, h1 = b1_x1 - b1_x0, b1_y1 - b1_y0 + eps
        w2, h2 = b2_x1 - b2_x0, b2_y1 - b2_y0 + eps

    inter_x0 = (b1_x1.minimum(b2_x1) - b1_x0.maximum(b2_x0)).clamp_(min=0)
    inter_y0 = (b1_y1.minimum(b2_y1) - b1_y0.maximum(b2_y0)).clamp_(min=0)
    inter_area = inter_x0 * inter_y0
    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    return inter_area / union_area
