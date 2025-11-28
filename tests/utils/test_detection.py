import pytest
import torch

from liandan.utils.detection import make_anchors


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
    assert anchors_tensor.shape == anchors_expected.shape
    assert strides_tensor.shape == strides_expected.shape
    assert anchors_tensor == pytest.approx(anchors_expected)
    assert strides_tensor == pytest.approx(strides_expected)
