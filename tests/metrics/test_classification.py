import pytest
import torch

from liandan.metrics.classification import Accuracy


# fmt: off
@pytest.mark.parametrize(
    "preds, targets, expected",
    [
        pytest.param(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), torch.tensor([1, 0]), 2 / 2, id="accuracy=2/2"),
        pytest.param(torch.tensor([[0.9, 0.1], [0.6, 0.4]]), torch.tensor([0, 1]), 1 / 2, id="accuracy=1/2"),
        pytest.param(torch.tensor([[0.2, 0.8], [0.4, 0.6]]), torch.tensor([0, 0]), 0 / 2, id="accuracy=0/2"),
        pytest.param(torch.empty((0, 2)), torch.empty((0,), dtype=torch.int64), 0.0, id="empty batch"),
    ],
)
def test_accuracy_parametrize(preds, targets, expected):
    acc = Accuracy()
    acc.update(preds, targets)
    assert acc.compute() == pytest.approx(expected)
# fmt: on


def test_accuracy_update_and_reset():
    acc = Accuracy()
    preds1 = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets1 = torch.tensor([0, 1])
    acc.update(preds1, targets1)
    # Updated 1st batch should get 2 corrects out of 2
    assert acc.compute() == 1.0

    preds2 = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
    targets2 = torch.tensor([1, 1])
    acc.update(preds2, targets2)
    # Updated 2nd batch should get 3 corrects out of 4
    assert acc.compute() == pytest.approx(3 / 4)

    acc.reset()
    assert acc.compute() == 0.0
