def as_pair[T](value: T | tuple[T, T]) -> tuple[T, T]:
    """將輸入值轉換為對應的 tuple。

    Args:
        value (T | tuple[T, T]): 輸入值。

    Returns:
        out (tuple[T, T]): 轉換後的 tuple。

    Raises:
        TypeError: 當輸入值類型不是`int`或`float`或其所組成的 tuple 時拋出。
    """
    match value:
        case (int(), int()):
            return value
        case (float(), float()):
            return value
        case int() | float():
            return value, value
        case _:
            raise TypeError(f"Unsupported type: {type(value)}")


def autopad(
    k: int | tuple[int, int],
    p: int | tuple[int, int] | None = None,
    d: int | tuple[int, int] = 1,
):
    """自動計算使卷積層輸出張量寬高不變所需要的 padding。

    Args:
        k (int | tuple[int, int]):
            Kernel size.
        p (int | tuple[int, int], optional):
            Padding. 預設值為`None`表示自動計算，否則直接回傳該值。
        d (int, optional):
            Dilation.

    Returns:
        out (int | tuple[int, int]): Padding.
    """
    k = as_pair(k)
    d = as_pair(d)
    k = (d[0] * (k[0] - 1) + 1, d[1] * (k[1] - 1) + 1)
    if p is None:
        p = (k[0] // 2, k[1] // 2)
    return p
