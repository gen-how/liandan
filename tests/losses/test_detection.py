import pytest
import torch

from liandan.losses.detection import TaskAlignedAssigner
from liandan.utils.detection import make_anchors


class TestTaskAlignedAssigner:
    @pytest.fixture
    def assigner(self):
        return TaskAlignedAssigner(num_classes=4)

    @pytest.fixture
    def anchors(self):
        anchor, stride = make_anchors([torch.zeros(20, 20)], [32])
        anchor_tensor = anchor * stride
        assert anchor_tensor.shape == (400, 2)
        return anchor_tensor

    @pytest.fixture
    def samples(self):
        r"""```
        ┌ Batch 0 ─────────┐  ┌ Batch 1 ─────────┐ Image Size: (640, 640)
        │                  │  │                  │ A: xyxy=[100, 100, 200, 200], cls=1
        │  ┌ A ┐           │  │    ┌ C ────────┐ │ B: xyxy=[300, 300, 450, 450], cls=3
        │  │   │           │  │    │           │ │ C: xyxy=[200, 100, 500, 400], cls=0
        │  └───┘           │  │    │           │ │
        │       ┌ B ─┐     │  │    │           │ │
        │       │    │     │  │    │           │ │
        │       │    │     │  │    └───────────┘ │
        │       └────┘     │  │                  │
        └──────────────────┘  └──────────────────┘
        ```"""
        gt_classes = torch.tensor(
            [
                [[1], [3]],  # Batch 0
                [[0], [0]],  # Batch 1
            ],
            dtype=torch.int64,
        )
        gt_boxes = torch.tensor(
            [
                [  # Batch 0
                    [100, 100, 200, 200],  # A
                    [300, 300, 450, 450],  # B
                ],
                [  # Batch 1
                    [200, 100, 500, 400],  # C
                    [0, 0, 0, 0],  # padded box
                ],
            ],
            dtype=torch.float32,
        )
        gt_mask = torch.tensor(
            [
                [[1], [1]],  # Batch 0
                [[1], [0]],  # Batch 1
            ],
            dtype=torch.bool,
        )
        return (gt_boxes, gt_classes, gt_mask)

    def test_select_anchors_in_gts_with_fixture(self, anchors, samples):
        gt_boxes, _, _ = samples
        expected = torch.zeros((2, 2, 20, 20), dtype=torch.bool)
        # anchors coordinates: 0.5 * stride + n * stride
        # A: 16 + 32 * 3 = 112, 16 + 32 * 6 = 208
        expected[0, 0, 3:6, 3:6] = True
        # B: 16 + 32 * 9 = 304, 16 + 32 * 14 = 464
        expected[0, 1, 9:14, 9:14] = True
        # C: 16 + 32 * 3 = 112, 16 + 32 * 12 = 400 (y-axis)
        #    16 + 32 * 6 = 208, 16 + 32 * 15 = 496 (x-axis)
        expected[1, 0, 3:12, 6:16] = True
        expected = expected.view(2, 2, 400)
        selected = TaskAlignedAssigner.select_anchors_in_gts(anchors, gt_boxes)
        torch.testing.assert_close(selected, expected)

    def test_select_anchors_in_gts(self):
        # Makes dummy anchors.
        sx = torch.arange(8, dtype=torch.float32) + 0.5
        sy = torch.arange(8, dtype=torch.float32) + 0.5
        gx, gy = torch.meshgrid(sx, sy, indexing="xy")
        anchors = torch.stack((gx, gy), dim=-1).view(-1, 2)
        assert anchors.shape == (64, 2)

        # Makes dummy gt_boxes
        gt_boxes = torch.tensor(
            [
                [  # Batch 0
                    [0.0, 0.0, 2.0, 2.0],  # Box 0
                    [0.0, 0.0, 0.0, 0.0],  # padded box
                ],
                [  # Batch 1
                    [7.0, 7.0, 9.0, 9.0],  # Box 1
                    [3.0, 3.0, 5.0, 5.0],  # Box 2
                ],
            ]
        )
        assert gt_boxes.shape == (2, 2, 4)

        # fmt: off
        expected = torch.tensor(
            [
                [  # Batch 0
                    [  # Box 0
                        1, 1, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [  # padded box
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ],
                [  # Batch 1
                    [  # Box 1
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 1,
                    ],
                    [  # Box 2
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 0, 0, 0,
                        0, 0, 0, 1, 1, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ],
            ], dtype=torch.bool
        )
        # fmt: on
        assert expected.shape == (2, 2, 64)
        torch.testing.assert_close(
            TaskAlignedAssigner.select_anchors_in_gts(anchors, gt_boxes), expected
        )
