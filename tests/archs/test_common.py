import pytest

from liandan.archs.common import as_pair, autopad


@pytest.mark.parametrize(
    "value, expected, exception",
    [
        pytest.param(3, (3, 3), None, id="as_pair(3)"),
        pytest.param(3.5, (3.5, 3.5), None, id="as_pair(3.5)"),
        pytest.param((3, 3), (3, 3), None, id="as_pair((3,3))"),
        pytest.param((3.5, 3.5), (3.5, 3.5), None, id="as_pair((3.5,3.5))"),
        pytest.param((3, 3, 3), None, TypeError, id="as_pair((3,3,3))"),
        pytest.param("3", None, TypeError, id="as_pair('3')"),
    ],
)
def test_as_pair(value, expected, exception):
    if exception:
        with pytest.raises(exception):
            as_pair(value)
    else:
        assert as_pair(value) == expected


@pytest.mark.parametrize(
    "k, p, d, expected",
    [
        pytest.param(3, None, 1, (1, 1), id="autopad(k=3)"),
        pytest.param(3, None, 2, (2, 2), id="autopad(k=3,d=2)"),
        pytest.param((3, 5), None, 1, (1, 2), id="autopad(k=(3,5))"),
        pytest.param((3, 5), None, (2, 1), (2, 2), id="autopad(k=(3,5),d=(2,1))"),
        pytest.param(3, 2, 1, 2, id="autopad(k=3,p=2)"),
    ],
)
def test_autopad(k, p, d, expected):
    assert autopad(k, p, d) == expected
