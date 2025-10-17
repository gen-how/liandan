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


def tensor_to_cv_mat(tensor: torch.Tensor, color_fmt="rgb"):
    r"""將 PyTorch 張量轉換為 OpenCV Mat 格式。

    Args:
        tensor: 形狀為 (C, H, W) 且 dtype 為 uint8 的 3 維張量。
        color_fmt: 輸入張量的顏色格式。可以是`"rgb"`或`"bgr"`。

    Returns:
        numpy.NDArray: 形狀為 (H, W, C) 且 dtype 為 uint8 的陣列。
    """
    if tensor.ndim != 3:
        raise ValueError("Input tensor must be 3-dimensional (C, H, W).")
    if tensor.dtype != torch.uint8:
        raise ValueError("Input tensor must have dtype uint8.")

    # If color format in Tensor is RGB, convert to BGR for OpenCV.
    if color_fmt == "rgb":
        tensor = tensor[[2, 1, 0], :, :]
    # Converts from (C, H, W) to (H, W, C).
    return tensor.permute(1, 2, 0).contiguous().numpy(force=True)
