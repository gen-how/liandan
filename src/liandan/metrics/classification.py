import torch

from .base import Metric


class Accuracy(Metric):
    """分類準確率。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = torch.tensor(0)
        # Tensor.size(0) 回傳的是 int, 不需要初始化成 Tensor
        self.total = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """更新正確預測數與總樣本數。

        Args:
            preds: 模型預測結果，形狀為 `(batch_size, num_classes)`。
            targets: 真實的類別索引，形狀為 `(batch_size,)`。
        """
        preds = torch.argmax(preds, dim=1)
        self.correct += (preds == targets).sum()
        self.total += targets.size(0)

    def compute(self):
        """計算並返回準確率。"""
        return (self.correct.float() / self.total) if self.total > 0 else 0.0
