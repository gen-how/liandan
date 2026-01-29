import pytest
import torch

from liandan.models._yolo import DFLConv, DFLLinear


@pytest.fixture(scope="module")
def reg_max():
    return 16


@pytest.fixture(scope="module")
def dummy_input(reg_max):
    # Gathered outputs from detection head.
    # (batch, 4 * reg_max, num_anchors)
    b, nc, na = (1, 4 * reg_max, 1)
    dummy = torch.zeros((b, nc, na), dtype=torch.float32)
    dummy.view(b, 4, reg_max, na)[:, :, 15, :] = 100.0
    return dummy


class TestDFLConv:
    @pytest.fixture(scope="class")
    def obj(self, reg_max):
        return DFLConv(reg_max=reg_max)

    def test_weight_init_and_frozen(self, obj, reg_max):
        expected = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max, 1, 1)
        torch.testing.assert_close(obj.conv.weight.detach(), expected)
        assert not obj.conv.weight.requires_grad

    def test_forward(self, obj, dummy_input):
        output = obj(dummy_input)
        b, _, na = dummy_input.shape
        expected = torch.full((b, 4, na), 15.0, dtype=torch.float32)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


class TestDFLLinear:
    @pytest.fixture(scope="class")
    def obj(self, reg_max):
        return DFLLinear(reg_max=reg_max)

    def test_weight_init_and_frozen(self, obj, reg_max):
        expected = torch.arange(reg_max, dtype=torch.float32).view(1, reg_max)
        torch.testing.assert_close(obj.fc.weight.detach(), expected)
        assert not obj.fc.weight.requires_grad

    def test_forward(self, obj, dummy_input):
        output = obj(dummy_input)
        b, _, na = dummy_input.shape
        expected = torch.full((b, na, 4), 15.0, dtype=torch.float32)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
