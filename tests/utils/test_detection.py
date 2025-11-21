import torch

from liandan.utils.detection import bbox_iou


def test_bbox_iou_xyxy_overlap_and_disjoint():
    box1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    box2 = torch.tensor([[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]])

    iou = bbox_iou(box1, box2)

    expected = torch.tensor([[1.0 / 7.0], [0.0]])
    torch.testing.assert_close(iou, expected, atol=1e-6, rtol=0.0)


def test_bbox_iou_xywh_broadcasting():
    box1 = torch.tensor([[[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]])
    box2 = torch.tensor([[[1.0, 1.0, 2.0, 2.0]]])

    iou = bbox_iou(box1, box2, xywh=True)

    expected = torch.tensor([[[1.0], [1.0 / 7.0]]])
    torch.testing.assert_close(iou, expected, atol=1e-6, rtol=0.0)
