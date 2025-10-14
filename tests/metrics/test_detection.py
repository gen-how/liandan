import pytest
import torch

from liandan.metrics.detection import AveragePrecision, MeanAveragePrecision


# fmt: off
@pytest.mark.parametrize(
    "iou_lower, iou_upper, iou_step, expected",
    [
        pytest.param(0.50, 1.0, 0.05, [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], id="AP@[IoU=0.50:0.95]",),
        pytest.param(0.50, 1.0, 1.00, [0.50], id="AP@[IoU=0.50]"),
        pytest.param(0.75, 1.0, 1.00, [0.75], id="AP@[IoU=0.75]"),
    ],
)
def test_average_precision_set_iou_ths(iou_lower, iou_upper, iou_step, expected):
    ap = AveragePrecision(iou_lower=iou_lower, iou_upper=iou_upper, iou_step=iou_step)
    assert ap.iou_ths == pytest.approx(expected)
# fmt: on


def test_average_precision_update_no_preds():
    ap = AveragePrecision()
    preds = torch.empty((0, 5))
    targets = torch.tensor([[0, 0, 10, 10, 0]])
    ap.update(preds, targets)
    assert ap.dts.shape == (0, 5)
    assert ap.gts.shape == (1, 5)
    assert ap.dts_iou.shape == (0,)


def test_average_precision_update_no_targets():
    ap = AveragePrecision()
    preds = torch.tensor([[0, 0, 10, 10, 0.9]])
    targets = torch.empty((0, 5))
    ap.update(preds, targets)
    assert ap.dts.shape == (1, 5)
    assert ap.gts.shape == (0, 5)
    assert ap.dts_iou.shape == (1,)
    assert ap.dts_iou[0] == 0.0


@pytest.mark.parametrize(
    "num_classes",
    [
        pytest.param(1, id="num_classes=1"),
        pytest.param(2, id="num_classes=2"),
        pytest.param(3, id="num_classes=3"),
    ],
)
def test_mean_average_precision_initialization(num_classes):
    map_ = MeanAveragePrecision(num_classes)
    assert map_.num_classes == len(map_.aps)
