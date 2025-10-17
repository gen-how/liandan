import torch


def get_default_device():
    """根據作業系統取得預設的加速計算裝置。

    Returns:
        torch.device: 代表計算裝置的物件。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
