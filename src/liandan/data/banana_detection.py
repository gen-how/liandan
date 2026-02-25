import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from torchvision.io import decode_image
from torchvision.tv_tensors import BoundingBoxes, Image

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

        self.images, self.labels = self._load_data()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img = Image(self.images[index])
        boxes = BoundingBoxes(
            self.labels[index, :, 1:],
            format="XYXY",
            canvas_size=(img.shape[-2], img.shape[-1]),
        )  # type: ignore
        sample = {
            "boxes": boxes,
            "classes": self.labels[index, :, 0],
            "image": img,
        }
        return self.transform(sample) if self.transform else sample

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
            return torch.stack(images), torch.tensor(labels).unsqueeze_(1)

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
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2 as T

    t = T.Compose(
        [
            T.RandomCrop(size=(64, 64)),
            T.ToDtype(torch.float32, scale=True),
        ]
    )
    bd = BananaDetection("./datasets/banana-detection", split="train", transform=t)
    dl = DataLoader(bd, batch_size=8, collate_fn=BananaDetection.collate_fn)

    # 獲取一個批次並可視化
    for batch in dl:
        images = batch["images"]
        boxes = batch["boxes"]
        classes = batch["classes"]

        batch_size = images.shape[0]
        # 創建 2x4 網格的大圖
        grid_h, grid_w = 2, 4
        img_h, img_w = 64, 64
        canvas = np.zeros((grid_h * img_h, grid_w * img_w, 3), dtype=np.uint8)

        print("==========")
        for i in range(batch_size):
            # 轉換為 numpy 並從 RGB 轉為 BGR（OpenCV 格式）
            img = images[i].permute(1, 2, 0).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 繪製偵測框
            for box, cls in zip(boxes[i], classes[i], strict=True):
                if cls >= 0:  # 只繪製有效類別的框
                    x0, y0, x1, y1 = box.tolist()
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    print(f"Image {i}, Box: ({x0}, {y0}, {x1}, {y1})")

            # 計算在網格中的位置
            row = i // grid_w
            col = i % grid_w
            y_start, y_end = row * img_h, (row + 1) * img_h
            x_start, x_end = col * img_w, (col + 1) * img_w

            canvas[y_start:y_end, x_start:x_end] = img

        # 顯示拼接後的圖片
        cv2.imshow("Batch Visualization", canvas)
        key = cv2.waitKey(0)
        if key == 27:  # 按下 ESC 鍵退出
            cv2.destroyAllWindows()
            break
