import csv
from pathlib import Path
from typing import Any, Literal

import torch
from torchvision.io import decode_image

from liandan.utils.data import calculate_md5, download_file, extract_zip


class BananaDetection(torch.utils.data.Dataset):
    """香蕉檢測資料集。

    此資料集用於訓練香蕉檢測模型，包含訓練集與驗證集兩個部分。
    此資料集取自李沐博士的教學系列影片 [41 物体检测和数据集【动手学深度学习v2】](https://www.bilibili.com/video/BV1Lh411Y7LX/?p=3)。
    """

    MIRRORS = ("http://d2l-data.s3-accelerate.amazonaws.com/",)
    RESOURCES = (("banana-detection.zip", "191823bdb3e62ff13738cc27fa5ee5dd"),)

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "valid"],
        download=False,
    ):
        """根據`split`載入不同部分的香蕉檢測資料集。

        Args:
            root (str | Path): 資料集的根目錄。
            split (str): 選擇載入哪一部分的資料集，必需是`"train"`或`"valid"`。
            download (bool, optional): 是否下載並解壓資料集，預設值為`False`。
        """
        self.root = Path(root).expanduser()
        self.split = split

        if download:
            self._download_and_extract()

        self.images, self.labels = self._load_data()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {
            "boxes": self.labels[index, 1:],
            "classes": self.labels[index, 0],
            "image": self.images[index],
        }
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """提供給`torch.utils.data.DataLoader`使用的批次整理函數。

        Args:
            batch (list[dict[str, Any]]): 單一批次的樣本列表。

        Returns:
            out (dict[str, torch.Tensor]): 整理後的批次資料。
        """
        collated = {
            "images": torch.stack([b["image"] for b in batch]),
            "batch_idx": torch.arange(len(batch)),
            "boxes": torch.stack([b["boxes"] for b in batch]),
            "classes": torch.stack([b["classes"] for b in batch]),
        }
        return collated

    def _load_data(self):
        split_name = {"train": "bananas_train", "valid": "bananas_val"}
        split_dir = self.root / split_name[self.split]
        # This is a small dataset, so we load all data into memory.
        with (split_dir / "label.csv").open("r") as f:
            reader = csv.reader(f.readlines())
            _ = next(reader)  # Skips header
            image_dir = split_dir / "images"
            images = []
            labels = []
            for row in reader:
                # Each row contains [img_name, cls, x0, y0, x1, y1].
                images.append(decode_image(str(image_dir / row[0])))
                labels.append(list(map(int, row[1:])))
            # All images have the same shape, so we can stack them directly.
            return torch.stack(images), torch.tensor(labels)

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
