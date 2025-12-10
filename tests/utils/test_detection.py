import pytest
import torch

from liandan.utils.detection import boxes_iou, ltrb2xyxy, make_anchors


def test_make_anchors():
    # Dummy features.
    features = [
        torch.zeros(1, 3, 3, 4),
        torch.zeros(1, 3, 2, 2),
        torch.zeros(1, 3, 1, 1),
    ]
    strides = (8, 16, 32)
    # fmt: off
    anchors_expected = torch.tensor(
        [
            # 3x4
            [0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5],
            [0.5, 1.5], [1.5, 1.5], [2.5, 1.5], [3.5, 1.5],
            [0.5, 2.5], [1.5, 2.5], [2.5, 2.5], [3.5, 2.5],
            # 2x2
            [0.5, 0.5], [1.5, 0.5],
            [0.5, 1.5], [1.5, 1.5],
            # 1x1
            [0.5, 0.5],
        ],
        dtype=torch.float32
    )
    strides_expected = torch.tensor(
        [
            # 4x3
            [8], [8], [8], [8],
            [8], [8], [8], [8],
            [8], [8], [8], [8],
            # 2x2
            [16], [16],
            [16], [16],
            # 1x1
            [32],
        ]
        ,dtype=torch.float32
    )
    # fmt: on
    anchors_tensor, strides_tensor = make_anchors(features, strides)
    torch.testing.assert_close(anchors_tensor, anchors_expected)
    torch.testing.assert_close(strides_tensor, strides_expected)


def test_ltrb2xyxy():
    # fmt: off
    # Dummy anchor points with shape: (4, 2).
    anchor_points = torch.tensor(
        [
            [ 5.0,  5.0],
            [10.0, 10.0],
            [15.0, 15.0],
            [20.0, 20.0],
        ]
    )
    # Dummy ltrb with shape: (2, 4, 4).
    # Because it comes from network output, there will be a batch dimension.
    ltrb = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
        ]
    )

    # Expects broadcasted result with shape: (2, 4, 4).
    expected = torch.tensor(
        [
            [
                [ 4.0,  4.0,  6.0,  6.0],
                [ 9.0,  9.0, 11.0, 11.0],
                [14.0, 14.0, 16.0, 16.0],
                [19.0, 19.0, 21.0, 21.0],
            ],
            [
                [ 3.0,  3.0,  7.0,  7.0],
                [ 8.0,  8.0, 12.0, 12.0],
                [13.0, 13.0, 17.0, 17.0],
                [18.0, 18.0, 22.0, 22.0],
            ],
        ]
    )
    # fmt: on
    result = ltrb2xyxy(ltrb, anchor_points)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "boxes1, boxes2, cxcywh, expected",
    [
        pytest.param(
            torch.tensor([0.0, 0.0, 5.0, 5.0]),
            torch.tensor([2.0, 2.0, 7.0, 7.0]),
            False,
            torch.tensor([9.0 / (25.0 + 25.0 - 9.0) + 1e-7]),
            id="boxes_iou(cxcywh=False) w/ shape (4,) & (4,)",
        ),
        pytest.param(
            torch.tensor([2.5, 2.5, 5.0, 5.0]),
            torch.tensor([4.5, 4.5, 5.0, 5.0]),
            True,
            torch.tensor([9.0 / (25.0 + 25.0 - 9.0) + 1e-7]),
            id="boxes_iou(cxcywh=True) w/ shape (4,) & (4,)",
        ),
        # TODO: Add more test cases for different shapes.
    ],
)
def test_boxes_iou(boxes1, boxes2, cxcywh, expected):
    result = boxes_iou(boxes1, boxes2, cxcywh)
    torch.testing.assert_close(result, expected)
