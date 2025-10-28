import cv2
import torch
from cv2.typing import MatLike, Point, Scalar

# fmt: off
G = (  0, 255,   0)
B = (255,   0,   0)
R = (  0,   0, 255)
Y = (  0, 255, 255)
P = (255,   0, 255)
W = (255, 255, 255)
# fmt: on


def from_tensor(tensor: torch.Tensor, color_fmt="rgb") -> MatLike:
    r"""將 PyTorch 張量轉換為 OpenCV Mat 格式。

    Args:
        tensor (Tensor): 形狀為 (C, H, W) 且 dtype 為 uint8 的 3 維張量。
        color_fmt (str): 輸入張量的顏色格式。可以是`"rgb"`或`"bgr"`。

    Returns:
        out (MatLike): 轉換後的 OpenCV Mat。
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


def get_text_size(
    text: str,
    font_scale=1.0,
    font_face=cv2.FONT_HERSHEY_COMPLEX,
    thickness=1,
):
    """取得文字在影像中的尺寸。

    Args:
        text (str): 欲測量的文字內容。
        font_face (int): 字型，預設值為`cv2.FONT_HERSHEY_COMPLEX`。
        font_scale (float): 字型大小比例，預設值為`1.0`。
        thickness (int): 文字粗細，預設值為`1`。

    Returns:
        out (tuple[int, int, int]): `(文字寬度, 文字高度, 基線高度)`。
    """
    (w, h), base = cv2.getTextSize(text, font_face, font_scale, thickness)
    return w, h, base


def get_font_scale(pixel_height: int, font_face=cv2.FONT_HERSHEY_COMPLEX, thickness=1):
    """根據指定的像素高度取得適當的字型縮放比例。

    Args:
        pixel_height (int): 目標文字的像素高度。
        font_face (int): 字型，預設值為`cv2.FONT_HERSHEY_COMPLEX`。
        thickness (int): 文字粗細，預設值為`1`。

    Returns:
        out (float): 適當的字型縮放比例。
    """
    return cv2.getFontScaleFromHeight(font_face, pixel_height, thickness)


def rectangle(
    img: MatLike,
    xyxy: list[int],
    color: Scalar,
    thickness=1,
    **kwargs,
):
    cv2.rectangle(img, xyxy[0:2], xyxy[2:4], color, thickness, **kwargs)


def text(
    img: MatLike,
    text: str,
    org: Point,
    color: Scalar,
    font_scale=1.0,
    font_face=cv2.FONT_HERSHEY_COMPLEX,
    thickness=1,
    **kwargs,
):
    cv2.putText(img, text, org, font_face, font_scale, color, thickness, **kwargs)


def text_autoscale(
    img: MatLike,
    text: str,
    org: Point,
    color: Scalar,
    font_face=cv2.FONT_HERSHEY_COMPLEX,
    thickness=1,
    move_base=False,
    **kwargs,
):
    pixel_height = max(round(img.shape[0] / 50), 12)
    scale = cv2.getFontScaleFromHeight(font_face, pixel_height, thickness)
    if move_base:
        _, base = cv2.getTextSize(text, font_face, scale, thickness)
        org = (org[0], org[1] - base)
    cv2.putText(img, text, org, font_face, scale, color, thickness, **kwargs)
