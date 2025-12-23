from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from liandan.utils.detection import boxes_ciou, ltrb2xyxy, make_anchors, xyxy2ltrb


class YOLOv8DetectionLoss(nn.Module):
    """搭配 YOLOv8 網路架構的物件偵測損失函式。"""

    def __init__(
        self, strides: Sequence[int], num_classes: int = 80, reg_max: int = 16
    ) -> None:
        """初始化 YOLOv8 物件偵測損失函式。

        Args:
            strides (Sequence[int]): 每個 Head 的錨點距離對應到輸入尺寸的實際步幅。
            num_classes (int, optional): 偵測類別數量。預設為`80`(COCO Dataset)。
            reg_max (int, optional): 使用多少個值來表示偵測框回歸分佈。預設為`16`。
        """
        super().__init__()
        # The strides is defined by the YOLOv8 architecture, and all YOLOv8 variants use
        # the same strides if I understand correctly.
        self.strides = strides
        self.nc = num_classes
        self.reg_max = reg_max
        self.nr = reg_max * 4
        self.no = self.nr + self.nc
        self.gain = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
        # The anchors depend on the output features of YOLOv8 detection heads.
        self.anchor_tensor = torch.empty(0)
        self.stride_tensor = torch.empty(0)
        self.img_hw = torch.zeros(2, dtype=torch.int64)
        self.assigner = TaskAlignedAssigner(num_classes=num_classes)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.dfl = DistributionFocalLoss(reg_max=reg_max)
        self.proj = nn.Buffer(torch.arange(self.reg_max, dtype=torch.float32))

    @property
    def device(self):
        # We can borrow the device attribute from any `nn.Buffer` or `nn.Parameter` in
        # this `nn.Module` to get the device where this module is located.
        return self.proj.device

    def forward(self, predict: list[torch.Tensor], targets: dict[str, torch.Tensor]):
        loss = torch.zeros(3, device=self.device)
        gt_boxes, gt_classes, gt_mask = self.targets_preprocess(targets)

        img_hw = torch.tensor(predict[0].shape[2:]) * self.strides[0]
        if (img_hw != self.img_hw).any():
            self.img_hw = img_hw
            self.anchor_tensor, self.stride_tensor = make_anchors(predict, self.strides)

        pd_scores, pd_distri = self.predict_preprocess(predict)
        pd_boxes = self.decode_bboxes(pd_distri, self.anchor_tensor)

        target_scores, target_boxes, fg_mask = self.assigner(
            pd_scores.detach().sigmoid(),
            pd_boxes.detach() * self.stride_tensor,
            self.anchor_tensor * self.stride_tensor,
            gt_classes,
            gt_boxes,
            gt_mask,
        )
        # The number of positive samples is different in each iteration, so
        # we should normalize the loss.
        target_scores_sum = max(target_scores.sum(), 1.0)

        # Classification loss.
        loss[1] = self.bce_loss(pd_scores, target_scores).sum() / target_scores_sum

        if fg_mask.any():
            # Shape: (batch, num_anchors, num_classes) -> (num_foreground, 1)
            weight = target_scores.sum(dim=-1)[fg_mask].unsqueeze(-1)

            # Complete IoU loss.
            ious = boxes_ciou(pd_boxes[fg_mask], target_boxes[fg_mask])
            loss[0] = ((1.0 - ious) * weight).sum() / target_scores_sum

            # Distribution Focal Loss.
            target_ltrb = xyxy2ltrb(
                target_boxes / self.stride_tensor, self.anchor_tensor
            )
            dfls = self.dfl(pd_distri[fg_mask], target_ltrb[fg_mask])
            loss[2] = (dfls * weight).sum() / target_scores_sum

        loss[0] *= self.gain["box"]
        loss[1] *= self.gain["cls"]
        loss[2] *= self.gain["dfl"]
        batch_size = pd_scores.shape[0]
        return loss * batch_size, loss.detach()

    def targets_preprocess(self, targets: dict[str, torch.Tensor]):
        """預處理訓練資料標註。

        不同張圖片中所標註的偵測框數量不盡相同，因此需要以下處理：
            1. 找出此批次所有圖片中最多偵測框的數量`num_max_boxes`。
            2. 建立形狀`(batch, num_max_boxes, 4)`的張量來存放偵測框座標，並以 0 填充。
            3. 建立遮罩張量來標示填充值與實際偵測框座標。

        Args:
            targets (dict[str, torch.Tensor]): 包含以下鍵值對的字典：
                - "batch_idx": 偵測框所屬的批次索引張量。
                - "boxes": 偵測框張量。
                - "classes": 偵測框類別張量。

        Returns:
            out (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                1. 已填充的偵測框座標張量，形狀為`(batch, num_max_boxes, 4)`。
                2. 已填充的偵測框類別張量，形狀為`(batch, num_max_boxes, 1)`。
                3. 選取實際資料的遮罩張量，形狀為`(batch, num_max_boxes, 1)`。
        """
        batch_idx = targets["batch_idx"]
        # The `batch_idx` should be orderly and consecutive, so we can use
        # `unique_consecutive` instead to avoid unnecessary sorting in `unique`.
        # Ref: https://docs.pytorch.org/docs/stable/generated/torch.unique.html
        _, counts = batch_idx.unique_consecutive(return_counts=True)
        batch_size = len(counts)
        max_counts = counts.max().item()
        gt_boxes = torch.zeros(batch_size, max_counts, 4)
        gt_classes = torch.zeros(batch_size, max_counts, 1)
        pos = 0
        for b in range(batch_size):
            cnt = counts[b].item()
            if cnt > 0:
                gt_boxes[b, :cnt, :] = targets["boxes"][pos : pos + cnt, :]
                gt_classes[b, :cnt, 0] = targets["classes"][pos : pos + cnt]
                pos += cnt
        # The sum of the coordinates of any valid box is always greater than zero.
        gt_mask = gt_boxes.sum(dim=2, keepdim=True).gt_(0)
        device = self.device
        return (
            gt_boxes.to(device),
            gt_classes.to(device),
            gt_mask.to(device, dtype=torch.bool),
        )

    def predict_preprocess(self, predict: list[torch.Tensor]):
        """整合多個 Heads 輸出並將偵測框的分類及回歸分佈分開存放。

        神經網路的每個 Head 輸出`(batch, C, H, W)`，其中`C = reg_max * 4 + num_classes`。
        假設輸入 640x640 影像，總共 80 類，則 3 個 Head 的輸出分別為：
        `(batch, 144, 80, 80)`, `(batch, 144, 40, 40)`, `(batch, 144, 20, 20)`。
        此函式將這些輸出整理成`(batch, 8400, 80)`的分類和`(batch, 8400, 64)`的回歸分佈。

        Args:
            predict (list[torch.Tensor]): 包含所有 Heads 輸出的列表。

        Returns:
            out (tuple[torch.Tensor, torch.Tensor]):
                1. 所有錨點的分類分數的張量，形狀為`(batch, num_anchors, num_classes)`。
                2. 所有錨點的回歸分佈的張量，形狀為`(batch, num_anchors, reg_max * 4)`。
        """
        stacked = torch.cat([x.view(x.shape[0], self.no, -1) for x in predict], dim=2)
        pred_distri, pred_scores = torch.split(stacked, [self.nr, self.nc], dim=1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        return pred_scores, pred_distri

    def decode_bboxes(self, pred_distri: torch.Tensor, anchor_points: torch.Tensor):
        """根據回歸分佈及錨點位置計算出偵測框座標（依然是 Head 大小，未還原輸入尺寸）。

        Args:
            pred_distri (torch.Tensor):
                回歸分佈張量，形狀為`(batch, num_anchors, reg_max * 4)`。
            anchor_points (torch.Tensor):
                錨點座標張量，形狀為`(num_anchors, 2)`。

        Returns:
            out (torch.Tensor): 偵測框座標張量，形狀為`(batch, num_anchors, 4)`。
        """
        # (batch_size, num_anchors, reg_max * 4)
        b, a, c = pred_distri.shape
        pred_distri = pred_distri.view(b, a, 4, c // 4).softmax(dim=3)
        # Distance to left, top, right, bottom. (batch_size, num_anchors, 4)
        pred_ltrb = torch.matmul(pred_distri, self.proj)
        return ltrb2xyxy(pred_ltrb, anchor_points)


class TaskAlignedAssigner(nn.Module):
    """訓練物件偵測神經網路的正負樣本分配器。"""

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
    ):
        """初始化 Task-Aligned Assigner。

        Args:
            topk (int, optional): 每個真實標註選取多少個預測框作為正樣本。預設為`13`。
            num_classes (int, optional): 偵測類別數量。預設為`80`(COCO Dataset)。
            alpha (float, optional): 原始公式中的 α 係數。預設為`1.0`。
            beta (float, optional): 原始公式中的 β 係數。預設為`6.0`。
            eps (float, optional): 防止除以零的極小值。預設為`1e-9`。
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_boxes: torch.Tensor,
        anchor_tensor: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_mask: torch.Tensor,
    ):
        """計算正負樣本分配結果。

        Args:
            pd_scores (torch.Tensor):
                偵測框分類張量，形狀為`(batch, num_anchors, num_classes)`。
            pd_boxes (torch.Tensor):
                偵測框張量，形狀為`(batch, num_anchors, 4)`。
            anchor_tensor (torch.Tensor):
                錨點座標張量，形狀為`(num_anchors, 2)`。
            gt_classes (torch.Tensor):
                標註框類別張量，形狀為`(batch, num_max_boxes, 1)`。
            gt_boxes (torch.Tensor):
                標註框座標張量，形狀為`(batch, num_max_boxes, 4)`。
            gt_mask (torch.Tensor):
                標註框遮罩張量，形狀為`(batch, num_max_boxes, 1)`。

        Returns:
            out (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                1. 目標偵測框分類張量，形狀為`(batch, num_anchors, num_classes)`。
                2. 目標偵測框座標張量，形狀為`(batch, num_anchors, 4)`。
                3. 前景錨點遮罩張量，形狀為`(batch, num_anchors)`。
        """
        num_max_boxes = gt_boxes.shape[1]
        if num_max_boxes == 0:
            return (
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_boxes),
                torch.zeros_like(pd_scores[..., 0]),
            )

        # Shape: (batch, num_max_boxes, num_anchors)
        anchors_mask = TaskAlignedAssigner.select_anchors_in_gts(
            anchor_tensor, gt_boxes
        )
        # Shape: (batch, num_max_boxes, num_anchors)
        align_metric, ious = self._compute_align_metric(
            pd_scores, pd_boxes, gt_classes, gt_boxes, anchors_mask
        )
        # Shape: (batch, num_max_boxes, num_anchors)
        topk_mask = self._select_topk_candidates(align_metric, gt_mask)
        positive_mask = anchors_mask & topk_mask
        target_indices, fg_mask, positive_mask = (
            TaskAlignedAssigner.filter_duplicate_assignments(positive_mask, ious)
        )
        # Shape: (batch, num_anchors, num_classes), (batch, num_anchors, 4)
        target_scores, target_boxes = self._gather_targets(
            gt_classes, gt_boxes, target_indices, fg_mask
        )

        # Normalizes targets.
        align_metric *= positive_mask
        # Shape: (batch, num_max_boxes, num_anchors) -> (batch, num_max_boxes, 1)
        max_align_metric = align_metric.amax(dim=-1, keepdim=True)
        max_ious = (ious * positive_mask).amax(dim=-1, keepdim=True)
        # Shape: (batch, num_max_boxes, num_anchors) -> (batch, num_anchors, 1)
        norm_align_metric = (
            ((align_metric * max_ious) / (max_align_metric + self.eps))
            .amax(-2)
            .unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric
        return target_scores, target_boxes, fg_mask

    def _compute_align_metric(
        self,
        pd_scores: torch.Tensor,
        pd_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        anchors_mask: torch.Tensor,
    ):
        """根據提供的預測和標註計算對齊指標。"""
        batch = gt_boxes.shape[0]  # b
        num_boxes = gt_boxes.shape[-2]  # nb
        num_anchors = pd_boxes.shape[-2]  # na

        scores = torch.zeros_like(anchors_mask, dtype=torch.float32)
        # Shape: (b,) -> (b, 1) -> (b, nb)
        indices_batch = torch.arange(batch).view(-1, 1).expand(-1, num_boxes)
        # Shape: (b, nb, 1) -> (b, nb)
        indices_class = gt_classes.squeeze(-1)
        # Advanced indexing: (b, na, num_classes) -> (b, nb, na)
        # Extracts and assigns selected: (b, nb, na) -> (num_select,)
        scores[anchors_mask] = pd_scores[indices_batch, :, indices_class][anchors_mask]

        ious = torch.zeros_like(anchors_mask, dtype=torch.float32)
        # Shape: (b, na, 4) -> (b, 1, na, 4) -> (b, nb, na, 4) -> (num_select, 4)
        pd_boxes = pd_boxes.unsqueeze(1).expand(-1, num_boxes, -1, -1)[anchors_mask]
        # Shape: (b, nb, 4) -> (b, nb, 1, 4) -> (b, nb, na, 4) -> (num_select, 4)
        gt_boxes = gt_boxes.unsqueeze(2).expand(-1, -1, num_anchors, -1)[anchors_mask]
        # Assigns (num_select, 1) -> (num_select,)
        ious[anchors_mask] = boxes_ciou(gt_boxes, pd_boxes).squeeze_(-1).clamp_(min=0)

        align_metric = scores.pow(self.alpha) * ious.pow(self.beta)
        return align_metric, ious

    def _select_topk_candidates(
        self, align_metric: torch.Tensor, gt_mask: torch.Tensor
    ):
        """選取每個真實標註框的前`topk`個對齊指標最高的預測框。"""
        # Shape: (batch, num_boxes, topk)
        _, topk_indices = torch.topk(align_metric, self.topk, dim=-1)
        # Shape: (batch, num_boxes, 1) -> (batch, num_boxes, topk)
        topk_mask = gt_mask.expand(-1, -1, self.topk)
        # Filters out dummy boxes.
        topk_indices.masked_fill_(~topk_mask, 0)
        counts = torch.zeros_like(align_metric, dtype=torch.uint8)
        ones = torch.ones_like(topk_indices[:, :, :1], dtype=torch.uint8)
        for k in range(self.topk):
            # Adds 1 to corresponding positions of `counts` according to the k-th
            # largest metric indices. Each iteration operates on the whole batch.
            counts.scatter_add_(dim=-1, index=topk_indices[:, :, k : k + 1], src=ones)
        # If any position is selected multiple times, it means it is a filtered
        # dummy boxes. Because we filtered `topk_indices` earlier.
        counts.masked_fill_(counts > 1, 0)
        # Shape: (batch, num_boxes, num_anchors)
        return counts.bool()

    def _gather_targets(
        self,
        gt_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        target_indices: torch.Tensor,
        fg_mask: torch.Tensor,
    ):
        """根據分配結果收集目標類別和偵測框。"""
        # Transforms `target_indices` to flattened indices for advanced indexing.
        batch, num_boxes = gt_boxes.shape[:2]  # b, nb
        num_anchors = target_indices.shape[1]  # na
        # Shape: (b,) -> (b, 1)
        batch_indices = torch.arange(batch, device=target_indices.device).unsqueeze_(-1)
        # Adds an offset from the batch index to obtain flattened indices while
        # keeping the same shape: (b, na).
        target_indices = target_indices + batch_indices * num_boxes

        # Gathers target boxes according to flattened indices.
        # Shape: (b, nb, 4) -> (b * nb, 4) -> (b, na, 4)
        target_boxes = gt_boxes.view(-1, 4)[target_indices]

        # Gathers target classes according to flattened indices.
        # Shape: (b, nb, 1) -> (b * nb, 1) -> (b, na, 1)
        target_classes = gt_classes.view(-1, 1)[target_indices]
        # Sometimes the negative samples are marked as -1.
        target_classes.clamp_(0)
        # Creates one-hot encoded target scores.
        target_scores = torch.zeros(
            (batch, num_anchors, self.num_classes),
            dtype=torch.int64,
            device=gt_classes.device,
        )
        target_scores.scatter_(dim=2, index=target_classes, value=1)
        # Filters out the scores of background anchors.
        fg_scores_mask = fg_mask.unsqueeze(-1).expand(-1, -1, self.num_classes)
        target_scores.masked_fill_(~fg_scores_mask, 0)
        return target_scores, target_boxes

    @staticmethod
    def select_anchors_in_gts(
        anchors: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-9
    ) -> torch.Tensor:
        """選取落在真實標註框內的錨點。

        此函式預期輸入參數已滿足以下條件：
        - `anchors`錨點座標為輸入圖片尺度，也就是與`gt_boxes`相同的尺度。
        - `gt_boxes`標註框座標格式為`(x0, y0, x1, y1)`。

        Args:
            anchors (torch.Tensor): 錨點張量，形狀為`(num_anchors, 2)`。
            gt_boxes (torch.Tensor): 真實標註框張量，形狀為`(batch, num_boxes, 4)`。
            eps (float, optional): 增加數值穩定性的極小值。預設為`1e-9`。

        Returns:
            out (torch.Tensor):
                表示錨點是否選中的遮罩張量，形狀為`(batch, num_boxes, num_anchors)`。
        """
        # Gets original sizes before reshaping.
        num_anchors = anchors.shape[0]  # na
        batch_size, num_boxes = gt_boxes.shape[:2]  # b, nb

        # Top-left points and bottom-right points.
        # Shape: (b, nb, 4) -> (b * nb, 1, 4) ─┬─> (b * nb, 1, 2)
        #                                      └─> (b * nb, 1, 2)
        lt, rb = gt_boxes.view(-1, 1, 4).chunk(2, dim=2)
        # Shape: (na, 2) -> (1, na, 2).
        anchors = anchors.view(1, -1, 2)
        # These shapes are meant to satisfy broadcast semantics.
        # Ref: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
        # Distances from anchor points to the four sides of ground truth boxes.
        # Shape: (b * nb, na, 2) ─┐
        #        (b * nb, na, 2) ─┴─> (b * nb, na, 4) -> (b, nb, na, 4)
        bbox_deltas = torch.cat((anchors - lt, rb - anchors), dim=2)
        bbox_deltas = bbox_deltas.view(batch_size, num_boxes, num_anchors, 4)
        # An anchor point is inside a ground truth box if all of its distances to the
        # four sides are greater than zero.
        return bbox_deltas.amin(dim=3).gt(eps)

    @staticmethod
    def filter_duplicate_assignments(positive_mask: torch.Tensor, ious: torch.Tensor):
        """過濾重複分配的錨點。

        當一個錨點被分配給多個標註框時，只應保留對齊指標最高的分配結果。
        即一個錨點只能學習來自一個標註框的資訊，而一個標註框可以分配給多個錨點。

        Args:
            positive_mask (torch.Tensor):
                選中正向錨點的遮罩張量，形狀為`(batch, num_boxes, num_anchors)`。
            ious (torch.Tensor):
                錨點與標註框的 IoU，形狀為`(batch, num_boxes, num_anchors)`。

        Returns:
            out (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                1. 每個錨點所分配的標註框索引張量，形狀為`(batch, num_anchors)`。
                2. 前景錨點遮罩張量，形狀為`(batch, num_anchors)`。
                3. 經過過濾後的正向錨點遮罩張量，形狀為`(batch, num_boxes, num_anchors)`
        """
        # Assumes b=batch, nb=num_boxes, na=num_anchors in the fllowing comments.
        # Merges all boxes' masks to get foreground masks.
        # Shape: (b, nb, na) -> (b, na)
        fg_mask = positive_mask.sum(dim=-2)
        if fg_mask.max() > 1:
            # At least one anchor is assigned to multiple ground truth boxes.
            num_boxes = positive_mask.shape[1]
            # Finds anchors with duplicate assignments.
            # Shape: (b, na) -> (b, 1, na) -> (b, nb, na)
            duplicate_mask = fg_mask.unsqueeze(1).gt(1).expand(-1, num_boxes, -1)
            # Finds the indices of boxes that have max IoU for each anchor.
            # Shape: (b, nb, na) -> (b, 1, na)
            max_iou_indices = ious.argmax(dim=1, keepdim=True)
            max_iou_mask = torch.zeros_like(positive_mask)
            max_iou_mask.scatter_(dim=1, index=max_iou_indices, value=True)
            # Updates masks to overwrite duplicate assignments.
            positive_mask = torch.where(duplicate_mask, max_iou_mask, positive_mask)
            fg_mask = positive_mask.sum(dim=-2)

        # Finds every anchor's assigned box index.
        # Shape: (b, nb, na) -> (b, na)
        target_indices = positive_mask.argmax(dim=-2)
        return target_indices, fg_mask.bool(), positive_mask


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL)。"""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_distri: torch.Tensor, target: torch.Tensor):
        # Makes sure target values are within [0, reg_max).
        target = target.clamp_(min=0, max=self.reg_max - 1 - 0.01)
        # Computes target lower bounds and upper bounds.
        # If distance is 2.3 -> 2 * 0.7 + 3 * 0.3 = 2.3
        tl = target.long()
        tu = tl + 1
        wl = tu - target
        wu = 1 - wl
        # Shape: (batch, num_anchors, reg_max * 4) -> (batch * num_anchors * 4, reg_max)
        pd_dist = pred_distri.view(-1, self.reg_max)
        cl = F.cross_entropy(pd_dist, tl.view(-1), reduction="none").view(tl.shape)
        cu = F.cross_entropy(pd_dist, tu.view(-1), reduction="none").view(tu.shape)
        return (cl * wl + cu * wu).mean(dim=-1, keepdim=True)


if __name__ == "__main__":
    from liandan.data import BananaDetection

    loss = YOLOv8DetectionLoss((8, 16, 32), num_classes=80, reg_max=16)

    valid_set = BananaDetection("./datasets/banana-detection", split="valid")
    valid_data = torch.utils.data.DataLoader(
        valid_set, batch_size=4, shuffle=False, collate_fn=BananaDetection.collate_fn
    )
    for samples in valid_data:
        gt_boxes, gt_classes, gt_mask = loss.targets_preprocess(samples)

        print(f"{gt_boxes.shape=}, {samples['boxes'].shape=}")
        print(f"{gt_classes.shape=}, {samples['classes'].shape=}")
        print(f"{gt_mask.shape=}, {gt_classes.shape=}")
        break

    print("ok.")
