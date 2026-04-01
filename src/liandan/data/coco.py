import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch


class CocoDetection(torch.utils.data.Dataset):
    """COCO 物件檢測資料集。

    此資料集用於載入 [COCO 官方資料集](https://cocodataset.org/)的物件檢測標注，包含訓練集與驗證集兩個部分。

    References:
        - [T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in ECCV, 2014.](https://arxiv.org/abs/1405.0312)
        - [COCO Dataset Official Website](https://cocodataset.org/)
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "valid"],
        transform: Callable | None = None,
        color_fmt: Literal["rgb", "bgr"] = "rgb",
    ) -> None:
        """根據`split`載入不同部分的 COCO 物件檢測資料集。

        預期的目錄結構：

        ```
        root/
        ├── images/
        │   ├── train2017/
        │   └── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
        ```

        Args:
            root (str | Path): 資料集的根目錄。
            split (str): 選擇載入哪一部分的資料集，必需是`"train"`或`"valid"`。
            transform (Callable, optional): 資料轉換函數，預設值為`None`。
            color_fmt (str, optional): 影像色彩格式，必需是`"rgb"`或`"bgr"`，預設值為`"rgb"`。
        """  # noqa: E501
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"'{self.root}' does not exist.")

        split_name = {"train": "train2017", "valid": "val2017"}
        if split not in split_name:
            raise ValueError("The argument 'split' must be 'train' or 'valid'.")
        self.split = split_name[split]

        self.image_dir = self.root / "images" / self.split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"'{self.image_dir}' does not exist.")

        ann_path = self.root / "annotations" / f"instances_{self.split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"'{ann_path}' does not exist.")
        self._load_annotation(ann_path)

        self.transform = transform
        self.imread_flag = (
            cv2.IMREAD_COLOR_RGB if color_fmt.lower() == "rgb" else cv2.IMREAD_COLOR
        )

    def _load_annotation(self, ann_path: Path):
        with ann_path.open("r") as f:
            coco: dict[str, Any] = json.load(f)

        # Transforms original COCO category_id to zero-indexed class numbers.
        cats = coco["categories"]
        cat_id_to_class = {cat["id"]: i for i, cat in enumerate(cats)}
        self.class_names = [cat["name"] for cat in cats]

        # Sorts and collects image filenames by image ID.
        img_id_to_filename = {e["id"]: e["file_name"] for e in coco["images"]}
        sorted_ids = sorted(img_id_to_filename.keys())
        sorted_filenames = [img_id_to_filename[sid] for sid in sorted_ids]
        self.image_filenames = np.array(sorted_filenames, dtype="U")

        img_id_to_bboxes = defaultdict(list)
        img_id_to_classes = defaultdict(list)
        img_id_to_iscrowd = defaultdict(list)
        for ann in coco["annotations"]:
            ann: dict[str, Any]
            img_id = ann["image_id"]
            iscrowd = ann.get("iscrowd", 0)
            cls = cat_id_to_class[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            img_id_to_bboxes[img_id].append(xyxy)
            img_id_to_classes[img_id].append(cls)
            img_id_to_iscrowd[img_id].append(iscrowd)

        # Preloads labels into one contiguous array with offsets by sample.
        bboxes_by_sample = []
        for sid in sorted_ids:
            if sid in img_id_to_bboxes:
                bboxes_by_sample.append(np.stack(img_id_to_bboxes[sid]))
            else:
                bboxes_by_sample.append(np.empty((0, 4), dtype=np.float32))
        counts_by_sample = [bboxes.shape[0] for bboxes in bboxes_by_sample]
        classes_by_sample = [img_id_to_classes[sid] for sid in sorted_ids]
        iscrowd_by_sample = [img_id_to_iscrowd[sid] for sid in sorted_ids]
        self.labels_offset = np.array([0, *counts_by_sample], dtype=np.int64).cumsum()
        self.bboxes = np.concatenate(bboxes_by_sample, axis=0)
        self.classes = np.concatenate(classes_by_sample)
        self.iscrowd = np.concatenate(iscrowd_by_sample)

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_path = self.image_dir / self.image_filenames[index]
        image = cv2.imread(str(img_path), self.imread_flag)
        assert image is not None, f"Failed to load image '{img_path}'."

        head = self.labels_offset[index]
        tail = self.labels_offset[index + 1]
        bboxes = self.bboxes[head:tail]
        classes = self.classes[head:tail]
        iscrowd = self.iscrowd[head:tail]

        sample = {
            "image": image,
            "bboxes": bboxes,
            "classes": classes,
            "iscrowd": iscrowd,
        }

        if self.transform:
            sample = self.transform(**sample)

        if len(sample["bboxes"]) > 0:
            bboxes_tensor = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
            classes_tensor = torch.as_tensor(sample["classes"], dtype=torch.int64)
            iscrowd_tensor = torch.as_tensor(sample["iscrowd"], dtype=torch.int64)
            sample["bboxes"] = bboxes_tensor.view(-1, 4)
            sample["classes"] = classes_tensor.view(-1, 1)
            sample["iscrowd"] = iscrowd_tensor.view(-1, 1)
        else:
            sample["bboxes"] = torch.empty((0, 4), dtype=torch.float32)
            sample["classes"] = torch.empty((0, 1), dtype=torch.int64)
            sample["iscrowd"] = torch.empty((0, 1), dtype=torch.int64)

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
        iscrowd_by_sample = [b["iscrowd"] for b in batch]
        num_obj_by_sample = torch.tensor([boxes.shape[0] for boxes in bboxes_by_sample])
        batch_idx = torch.repeat_interleave(torch.arange(len(batch)), num_obj_by_sample)
        bboxes = torch.cat(bboxes_by_sample, dim=0)
        classes = torch.cat(classes_by_sample, dim=0)
        iscrowd = torch.cat(iscrowd_by_sample, dim=0)
        return {
            "images": images,
            "boxes": bboxes,
            "classes": classes,
            "iscrowd": iscrowd,
            "batch_idx": batch_idx,
        }


if __name__ == "__main__":
    import albumentations as A
    from torch.utils.data import DataLoader

    from liandan.utils.opencv import from_tensor, rectangle, text_autoscale

    t = A.Compose(
        [
            A.LongestMaxSize(max_size=640),
            A.PadIfNeeded(min_height=640, min_width=640),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            coord_format="pascal_voc",
            label_fields=["classes"],
        ),
    )
    ds = CocoDetection("./datasets/coco", split="valid", transform=t)
    dl = DataLoader(ds, batch_size=4, collate_fn=CocoDetection.collate_fn)

    # Visualizes each batch in a 2x2 grid. Press ESC or q to exit.
    for batch in dl:
        images = batch["images"]
        boxes = batch["boxes"]
        classes = batch["classes"]
        iscrowd = batch["iscrowd"]
        batch_idx = batch["batch_idx"]

        batch_size = images.shape[0]
        grid_h, grid_w = 2, 2
        img_h, img_w = images.shape[-2:]
        canvas = np.zeros((grid_h * img_h, grid_w * img_w, 3), dtype=np.uint8)

        for i in range(batch_size):
            img = from_tensor(images[i])

            sample_mask = batch_idx == i
            sample_boxes = boxes[sample_mask]
            sample_classes = classes[sample_mask]
            sample_iscrowd = iscrowd[sample_mask]

            for box, cls, crowd in zip(
                sample_boxes, sample_classes, sample_iscrowd, strict=True
            ):
                xyxy = box.int().tolist()
                rectangle(img, xyxy, (0, 0, 255) if crowd else (0, 255, 255))
                cls_name = ds.class_names[cls.item()]
                text_autoscale(
                    img,
                    cls_name,
                    xyxy[0:2],
                    (255, 0, 0),
                    move_base=True,
                )

            row = i // grid_w
            col = i % grid_w
            y0, y1 = row * img_h, (row + 1) * img_h
            x0, x1 = col * img_w, (col + 1) * img_w
            canvas[y0:y1, x0:x1] = img

        cv2.imshow("CocoDetection Batch", canvas)
        key = cv2.waitKey(0)
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
