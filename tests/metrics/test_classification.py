import pytest
import torch

from liandan.metrics.classification import Accuracy


def test_perfect_accuracy():
    acc = Accuracy()
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    acc.update(preds, targets)
    assert acc.compute() == 1.0


def test_partial_accuracy():
    acc = Accuracy()
    preds = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]])
    targets = torch.tensor([0, 1, 1])
    acc.update(preds, targets)
    # `torch.argmax(preds, dim=1)` -> `[0, 1, 0]` so 2 correct out of 3
    assert acc.compute() == pytest.approx(2 / 3)


def test_accumulation_and_reset():
    acc = Accuracy()
    preds1 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets1 = torch.tensor([0, 1])
    acc.update(preds1, targets1)
    # first batch: 2 correct out of 2
    assert acc.compute() == 1.0

    preds2 = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    targets2 = torch.tensor([1, 1])
    acc.update(preds2, targets2)
    # second batch: 1 correct -> total 3 correct out of 4
    assert acc.compute() == pytest.approx(3 / 4)

    acc.reset()
    # after reset the metric should report 0.0 (0/1 per implementation guard)
    assert acc.compute() == 0.0


def test_empty_batch_no_error():
    acc = Accuracy()
    preds = torch.empty((0, 10))
    targets = torch.empty((0,), dtype=torch.long)
    acc.update(preds, targets)
    # Should not raise and should keep accuracy at 0.0
    assert acc.compute() == 0.0
