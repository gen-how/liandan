import torch


def get_default_device():
    """根據作業系統取得預設的加速計算裝置。

    Returns:
        out (torch.device): 代表計算裝置的物件。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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
