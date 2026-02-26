from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torchvision.io import decode_image
from torchvision.tv_tensors import BoundingBoxes, Image

from liandan.utils import trasient_print


class KAIST8(torch.utils.data.Dataset):
    """行人檢測資料集。

    此資料集是經過資料清洗的 KAIST 行人檢測資料集的子集，包含訓練集與驗證集兩個部分。

    References:
        - S. Hwang, J. Park, N. Kim, Y. Choi and I. S. Kweon, "Multispectral pedestrian detection: Benchmark dataset and baseline," 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, USA, 2015, pp. 1037-1045, doi: 10.1109/CVPR.2015.7298706, url: https://ieeexplore.ieee.org/document/7298706
    """  # noqa: E501

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "valid"],
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"The path '{self.root}' does not exist.")

        split_alias = {"train": "train", "valid": "test"}
        if split not in split_alias:
            raise ValueError("The argument 'split' must be 'train' or 'valid'.")
        self.split = split_alias[split]
        self.transform = transform

        self.visible_dir = self.root / "visible" / self.split
        self.infrared_dir = self.root / "infrared" / self.split
        self.labels_dir = self.root / "labels" / self.split
        for path in (self.visible_dir, self.infrared_dir, self.labels_dir):
            if not path.exists():
                raise FileNotFoundError(f"The path '{path}' does not exist.")

        visible_ids = {p.stem for p in self.visible_dir.glob("*.jpg")}
        infrared_ids = {p.stem for p in self.infrared_dir.glob("*.jpg")}
        if visible_ids != infrared_ids:
            raise ValueError("The visible and infrared images do not match.")
        self.sample_ids = np.array(sorted(visible_ids), dtype="U")

        # Preloads labels into one contiguous array with offsets by sample.
        labels_by_sample = [self._load_label(sid) for sid in self.sample_ids]
        counts_by_sample = [labels.shape[0] for labels in labels_by_sample]
        self.labels_offset = np.array([0, *counts_by_sample], dtype=np.int64).cumsum()
        self.labels = np.concatenate(labels_by_sample, axis=0)
        trasient_print()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        visible_img = Image(decode_image(str(self.visible_dir / f"{sample_id}.jpg")))
        infrared_img = Image(decode_image(str(self.infrared_dir / f"{sample_id}.jpg")))
        img_h, img_w = visible_img.shape[-2:]
        head = self.labels_offset[index]
        tail = self.labels_offset[index + 1]
        classes = torch.as_tensor(self.labels[head:tail, 0:1], dtype=torch.int64)
        boxes = BoundingBoxes(
            torch.as_tensor(self.labels[head:tail, 1:5], dtype=torch.float32),
            format="XYXY",
            canvas_size=(img_h, img_w),
        )  # type: ignore

        sample: dict[str, Any] = {
            "visible": visible_img,
            "infrared": infrared_img,
            "boxes": boxes,
            "classes": classes,
        }
        return self.transform(sample) if self.transform else sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        classes_by_sample = [sample["classes"] for sample in batch]
        num_obj_by_sample = torch.tensor([cls.shape[0] for cls in classes_by_sample])
        batch_idx = torch.repeat_interleave(torch.arange(len(batch)), num_obj_by_sample)
        visible = torch.stack([sample["visible"] for sample in batch])
        infrared = torch.stack([sample["infrared"] for sample in batch])
        boxes = torch.cat([sample["boxes"] for sample in batch], dim=0)
        classes = torch.cat(classes_by_sample, dim=0)

        return {
            "visible": visible,
            "infrared": infrared,
            "batch_idx": batch_idx,
            "boxes": boxes,
            "classes": classes,
        }

    def _load_label(self, sample_id: str) -> np.ndarray:
        label_txt = self.labels_dir / f"{sample_id}.txt"
        trasient_print(f"Loading label from '{label_txt}'...")
        if label_txt.stat().st_size == 0:
            return np.empty((0, 5), dtype=np.float32)

        data = np.loadtxt(label_txt, dtype=np.float32, ndmin=2)
        if data.shape[1] != 5:
            raise ValueError(f"Invalid label format in '{label_txt}'.")

        # Transforms from (class, cx, cy, w, h) to (class, x0, y0, x1, y1).
        visible_img = decode_image(str(self.visible_dir / f"{sample_id}.jpg"))
        img_h, img_w = visible_img.shape[-2:]
        img_wh = np.array([img_w, img_h], dtype=np.float32)
        cxcy = data[:, 1:3]
        wh_2 = data[:, 3:5] * 0.5
        x0y0 = (cxcy - wh_2) * img_wh
        x1y1 = (cxcy + wh_2) * img_wh
        np.clip(x0y0, 0, img_wh, out=data[:, 1:3])
        np.clip(x1y1, 0, img_wh, out=data[:, 3:5])
        return data


if __name__ == "__main__":
    import cv2

    from liandan.utils.opencv import from_tensor, rectangle

    root = Path("./datasets/KAIST8")
    dataset = KAIST8(root=root, split="valid")
    for sample in dataset:
        visible_img: torch.Tensor = sample["visible"]
        infrared_img: torch.Tensor = sample["infrared"]
        classes: torch.Tensor = sample["classes"]
        boxes: torch.Tensor = sample["boxes"]

        for i, img in enumerate((visible_img, infrared_img)):
            img_np = from_tensor(img)
            for _, box in zip(classes, boxes, strict=True):
                xyxy = box.int().tolist()
                rectangle(img_np, xyxy, color=(0, 255, 0))

            cv2.imshow(f"debug_{i}", img_np)
        if cv2.waitKey(0) == 27:
            break
