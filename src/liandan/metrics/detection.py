import torch
from torchvision.ops import box_iou

from .base import Metric


class AveragePrecision(Metric):
    """計算單一類別的 Average Precision (AP)。"""

    def __init__(
        self,
        iou_lower=0.5,
        iou_upper=1.0,
        iou_step=0.05,
        max_preds=20000,
    ) -> None:
        """初始化 AveragePrecision。

        參數預設值表示計算 `IoU=0.50:0.95` 區間，若只想計算單一 IoU 閾值：

        >>> ap = AveragePrecision(iou_lower=0.50, iou_step=1.0)  # AP@[IoU=0.50]
        >>> ap = AveragePrecision(iou_lower=0.75, iou_step=1.0)  # AP@[IoU=0.75]

        Args:
            iou_lower (float): 計算 AP 時的最低 IoU 閾值。
            iou_upper (float): 計算 AP 時的最高 IoU 閾值。
            iou_step (float): `[iou_lower, iou_upper)` 區間的步長。
            max_preds (int): 納入計算的預測框數量。
        """
        self.iou_ths = torch.arange(iou_lower, iou_upper, iou_step)
        self.max_preds = max_preds
        self.reset()

    def reset(self):
        self.is_collated = False
        # [x0, y0, x1, y1, confidence]
        self.dts = torch.empty((0, 5), dtype=torch.float32)
        # [x0, y0, x1, y1, class]
        self.gts = torch.empty((0, 5), dtype=torch.float32)
        self.dts_iou = torch.empty((0,), dtype=torch.float32)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.is_collated = False
        self.dts = torch.vstack((self.dts, preds))
        self.gts = torch.vstack((self.gts, targets))

        # 如果沒有偵測框，不用做其他處理。
        if preds.shape[0] == 0:
            return

        # 如果沒有標註框，所有偵測框都當作 FP。
        if targets.shape[0] == 0:
            self.dts_iou = torch.cat((self.dts_iou, torch.zeros((preds.shape[0],))))
            return

        # 計算所有偵測框與所有標註框的 IoU
        iou_matrix = box_iou(preds[:, 0:4], targets[:, 0:4])

        # 貪婪搜尋匹配的偵測框與標註框
        iou_flat = iou_matrix.ravel()
        sorted_idxs = torch.argsort(iou_flat, descending=True)
        dt_idxs, gt_idxs = torch.unravel_index(sorted_idxs, iou_matrix.shape)

        # 確保每個標註框只匹配一個偵測框
        matched_dts = set()
        matched_gts = set()
        dts_iou = torch.zeros((preds.shape[0],))
        for d, g in zip(dt_idxs.tolist(), gt_idxs.tolist(), strict=True):
            # 若所有標註框都找到匹配，則剩餘的預測框都是 FP
            if len(matched_gts) == targets.shape[0]:
                break
            # 偵測框與任意標註框的 IoU 若小於 IoU 閾值可直接視為 FP
            iou = iou_matrix[d, g]
            if iou < self.iou_ths[0]:
                break
            # 找到匹配，將被挑選的偵測框與標註框記錄下來
            if d not in matched_dts and g not in matched_gts:
                dts_iou[d] = iou
                matched_dts.add(d)
                matched_gts.add(g)
        self.dts_iou = torch.cat((self.dts_iou, dts_iou))
        return

    def compute(self):
        if not self.is_collated:
            # 根據偵測框 confidence 重新排序
            order = torch.argsort(self.dts[:, 4], descending=True)
            self.dts = self.dts[order]
            self.dts_iou = self.dts_iou[order]
            self.rank = torch.arange(1, self.dts.shape[0] + 1)
            self.is_collated = True

        result = torch.zeros_like(self.iou_ths)
        for idx, iou_th in enumerate(self.iou_ths):
            tps = self.dts_iou > iou_th
            acc_tps = tps.cumsum(dim=0)
            precisions = acc_tps / self.rank
            recalls = acc_tps / self.gts.shape[0]

            # 將 precisions 變為單調非遞增
            for i in range(precisions.shape[0] - 1, 0, -1):
                if precisions[i] > precisions[i - 1]:
                    precisions[i - 1] = precisions[i]

            # 計算 Average Precision
            num_pts = min(precisions.shape[0], self.max_preds)
            dr = recalls[1 : num_pts - 0] - recalls[0 : num_pts - 1]
            result[idx] = torch.sum(precisions[0 : num_pts - 1] * dr)
        return result.mean().item()


def naive_ap_integral(precisions, recalls, num_pts=20000):
    ap = 0.0
    i = 0
    while i < num_pts - 1:
        p = precisions[i].item()
        r0 = recalls[i].item()
        for j in range(i + 1, num_pts):
            if p != precisions[j].item():
                r1 = recalls[j].item()
                i = j
                ap += p * (r1 - r0)
                break
    return ap


class MeanAveragePrecision(Metric):
    def __init__(
        self,
        num_classes: int,
        iou_lower=0.5,
        iou_upper=1.0,
        iou_step=0.05,
        max_preds=20000,
    ) -> None:
        self.num_classes = num_classes
        self.aps = [
            AveragePrecision(iou_lower, iou_upper, iou_step, max_preds)
            for _ in range(num_classes)
        ]
        self.reset()

    def reset(self):
        for ap in self.aps:
            ap.reset()

    def update(self, preds: torch.Tensor, idxs: torch.Tensor, targets: torch.Tensor):
        for i in range(self.num_classes):
            self.aps[i].update(preds[idxs == i], targets[targets[:, 4] == i])

    def compute(self):
        result = torch.zeros((self.num_classes,), dtype=torch.float32)
        for i in range(self.num_classes):
            result[i] = self.aps[i].compute()
        return result.mean().item()

