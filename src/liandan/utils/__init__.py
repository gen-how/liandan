from collections.abc import Callable
from typing import Protocol, Self


class _Module[**P, R](Protocol):
    """Protocol allowing us to unwrap `forward` method signatures.

    Ref:
        - https://github.com/pytorch/pytorch/issues/74746#issuecomment-3597468963
        - https://github.com/pytorch/pytorch/issues/74746#issuecomment-3600066341
    """

    def __call__(self: Self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def forward(self: Self, *args: P.args, **kwargs: P.kwargs) -> R: ...


def typed_module[**P, R](m: _Module[P, R]) -> Callable[P, R]:
    """回傳提供的模組，並保留型別提示。

    建議只在開發時使用此函式，完成後將其移除避免造成其他人的困惑*。

    Args:
        m: 欲保留型別提示的`torch.nn.Module`子類別實例。

    Returns:
        out: 未改變的實例`m`。
    """
    return m


def unwrap[T](optional: T | None) -> T:
    """嘗試展開`optional`物件，若失敗則拋出例外。

    Args:
        optional (T | None): 欲展開的物件。

    Returns:
        out (T): 展開後的物件。

    Raises:
        ValueError: 展開失敗時拋出。
    """
    if optional is None:
        raise ValueError("Failed to unwrap.")
    return optional
