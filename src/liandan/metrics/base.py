from abc import ABC, abstractmethod


class Metric(ABC):
    """儲存並計算各種評估指標的基礎類別。"""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass


class AverageLoss(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0

    def update(self, loss_value):
        self.total_loss += loss_value
        self.count += 1

    def compute(self):
        return self.total_loss / max(self.count, 1)
