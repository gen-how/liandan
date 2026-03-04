import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import albumentations as A
import cv2
import numpy as np
import torch

from liandan.utils.data import calculate_md5, download_file, extract_zip


class BananaDetection(torch.utils.data.Dataset):
    """香蕉檢測資料集。

    此資料集用於訓練香蕉檢測模型，包含訓練集與驗證集兩個部分，取自李沐博士的教學系列影片 [41 物体检测和数据集【动手学深度学习v2】](https://www.bilibili.com/video/BV1Lh411Y7LX/?p=3)。
    """  # noqa: E501

    MIRRORS = ("http://d2l-data.s3-accelerate.amazonaws.com/",)
    RESOURCES = (("banana-detection.zip", "191823bdb3e62ff13738cc27fa5ee5dd"),)

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "valid"],
        transform: Callable | None = None,
        download=False,
    ):
        """根據`split`載入不同部分的香蕉檢測資料集。

        Args:
            root (str | Path): 資料集的根目錄。
            split (str): 選擇載入哪一部分的資料集，必需是`"train"`或`"valid"`。
            transform (Callable, optional): 資料轉換函數，預設值為`None`。
            download (bool, optional): 是否下載並解壓資料集，預設值為`False`。
        """
        self.root = Path(root).expanduser()
        self.split = split
        self.transform = transform

        if download:
            self._download_and_extract()

        self._load_data()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img = self.images[index]
        head = self.labels_offset[index]
        tail = self.labels_offset[index + 1]
        bboxes = self.labels[head:tail, 1:]
        classes = self.labels[head:tail, :1]
        sample = {
            "image": img,
            "bboxes": bboxes,
            "classes": classes,
        }

        if self.transform:
            sample = self.transform(**sample)

        if len(sample["bboxes"]) > 0:
            bboxes_tensor = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            classes_tensor = torch.as_tensor(sample["classes"], dtype=torch.int64)
            sample["bboxes"] = bboxes_tensor.view(-1, 4)
            sample["classes"] = classes_tensor.view(-1, 1)
        else:
            sample["bboxes"] = torch.empty((0, 4), dtype=torch.float32)
            sample["classes"] = torch.empty((0, 1), dtype=torch.int64)

        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """提供給`torch.utils.data.DataLoader`使用的批次整理函數。

        Args:
            batch (list[dict[str, Any]]): 單一批次的樣本列表。

        Returns:
            out (dict[str, torch.Tensor]): 整理後的批次資料。
        """
        images = torch.stack([b["image"] for b in batch])
        bboxes_by_sample = [b["bboxes"] for b in batch]
        classes_by_sample = [b["classes"] for b in batch]
        num_obj_by_sample = torch.tensor([boxes.shape[0] for boxes in bboxes_by_sample])
        batch_idx = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.int64), num_obj_by_sample
        )
        bboxes = torch.cat(bboxes_by_sample, dim=0)
        classes = torch.cat(classes_by_sample, dim=0)
        return {
            "images": images,
            "boxes": bboxes,
            "classes": classes,
            "batch_idx": batch_idx,
        }

    def _load_data(self):
        split_name = {"train": "bananas_train", "valid": "bananas_val"}
        split_dir = self.root / split_name[self.split]
        # This is a small dataset, so we load all data into memory.
        with (split_dir / "label.csv").open("r") as f:
            reader = csv.reader(f.readlines())
        # Skips the header row.
        _ = next(reader)

        image_dir = split_dir / "images"
        images = []
        labels = []
        for row in reader:
            # Each row contains [img_name, cls, x0, y0, x1, y1].
            image_path = image_dir / row[0]
            image = cv2.imread(str(image_path))
            assert image is not None, f"Failed to load image '{image_path}'."
            images.append(image)
            labels.append(np.fromiter(row[1:], dtype=np.int64))

        # All images have the same shape, so we can stack them directly.
        self.images = np.stack(images)
        # Each image have exactly one bounding box.
        self.labels_offset = np.arange(len(labels) + 1)
        self.labels = np.stack(labels)

    def _download_and_extract(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for filename, md5 in BananaDetection.RESOURCES:
            filepath = self.root / filename
            # Checks if the resources is already downloaded.
            if not filepath.exists() or calculate_md5(filepath) != md5:
                # Downloads the resource from mirrors.
                for mirror in BananaDetection.MIRRORS:
                    url = f"{mirror}{filename}"
                    print(f"Downloading '{url}'...")
                    try:
                        download_file(url, filepath)
                    except Exception as e:
                        print(f"Failed to download from '{url}': {e}")
                        continue
                    if calculate_md5(filepath) == md5:
                        print(f"Successfully downloaded '{filename}'.")
                        break
                else:
                    raise RuntimeError(f"Failed to download '{filename}'.")
                # Extracts the downloaded resource.
                extract_zip(filepath, self.root.parent)
                print(f"Extracted '{filename}'.")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from liandan.utils.opencv import from_tensor, rectangle, text_autoscale

    t = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["classes"]),
    )
    bd = BananaDetection("./datasets/banana-detection", split="train", transform=t)
    dl = DataLoader(bd, batch_size=8, collate_fn=BananaDetection.collate_fn)

    # Visualizes each batch in a 2x4 grid. Press ESC or q to exit.
    for batch in dl:
        images = batch["images"]
        boxes = batch["boxes"]
        classes = batch["classes"]
        batch_idx = batch["batch_idx"]

        batch_size = images.shape[0]
        grid_h, grid_w = 2, 4
        img_h, img_w = images.shape[-2:]
        canvas = np.zeros((grid_h * img_h, grid_w * img_w, 3), dtype=np.uint8)

        for i in range(batch_size):
            img = from_tensor(images[i], color_fmt="bgr")

            sample_mask = batch_idx == i
            sample_boxes = boxes[sample_mask]
            sample_classes = classes[sample_mask]

            for box, cls in zip(sample_boxes, sample_classes, strict=True):
                xyxy = box.int().tolist()
                rectangle(img, xyxy, (0, 0, 255), thickness=2)
                text_autoscale(
                    img,
                    f"{cls.item()}",
                    xyxy[0:2],
                    (0, 255, 0),
                    move_base=True,
                )

            row = i // grid_w
            col = i % grid_w
            y0, y1 = row * img_h, (row + 1) * img_h
            x0, x1 = col * img_w, (col + 1) * img_w
            canvas[y0:y1, x0:x1] = img

        cv2.imshow("BananaDetection Batch", canvas)
        key = cv2.waitKey(0)
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
