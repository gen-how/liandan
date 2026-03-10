from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch
from PIL import Image

# from torchvision.io import decode_image
# from torchvision.tv_tensors import BoundingBoxes, Image


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
            raise FileNotFoundError(f"'{self.root}' does not exist.")

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
                raise FileNotFoundError(f"'{path}' does not exist.")

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

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        visible_img = cv2.imread(str(self.visible_dir / f"{sample_id}.jpg"))
        infrared_img = cv2.imread(str(self.infrared_dir / f"{sample_id}.jpg"))
        head = self.labels_offset[index]
        tail = self.labels_offset[index + 1]
        bboxes = self.labels[head:tail, 1:5]
        classes = self.labels[head:tail, 0:1]

        sample = {
            "image": visible_img,
            "ir_image": infrared_img,
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
        visible_images = torch.stack([b["image"] for b in batch])
        infrared_images = torch.stack([b["ir_image"] for b in batch])
        bboxes_by_sample = [b["bboxes"] for b in batch]
        classes_by_sample = [b["classes"] for b in batch]
        num_obj_by_sample = torch.tensor([boxes.shape[0] for boxes in bboxes_by_sample])
        batch_idx = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.int64), num_obj_by_sample
        )
        bboxes = torch.cat(bboxes_by_sample, dim=0)
        classes = torch.cat(classes_by_sample, dim=0)

        return {
            "images": visible_images,
            "ir_images": infrared_images,
            "boxes": bboxes,
            "classes": classes,
            "batch_idx": batch_idx,
        }

    def _load_label(self, sample_id: str) -> np.ndarray:
        label_txt = self.labels_dir / f"{sample_id}.txt"
        if label_txt.stat().st_size == 0:
            return np.empty((0, 5), dtype=np.float32)

        data = np.loadtxt(label_txt, dtype=np.float32, ndmin=2)
        if data.shape[1] != 5:
            raise ValueError(f"Invalid label format in '{label_txt}'.")

        # Transforms from (class, cx, cy, w, h) to (class, x0, y0, x1, y1).
        # Uses PIL `Image.open` lazy loading instead of `cv2.imread`.
        visible_img = Image.open(str(self.visible_dir / f"{sample_id}.jpg"))
        img_wh = np.array([visible_img.width, visible_img.height], dtype=np.float32)
        cxcy = data[:, 1:3]
        wh_2 = data[:, 3:5] * 0.5
        x0y0 = (cxcy - wh_2) * img_wh
        x1y1 = (cxcy + wh_2) * img_wh
        np.clip(x0y0, 0, img_wh, out=data[:, 1:3])
        np.clip(x1y1, 0, img_wh, out=data[:, 3:5])
        return data


if __name__ == "__main__":
    import albumentations as A
    from torch.utils.data import DataLoader

    from liandan.utils.opencv import from_tensor, rectangle, text_autoscale

    t = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["classes"]),
        additional_targets={"ir_image": "image"},
    )

    ds = KAIST8(root="./datasets/KAIST8", split="valid", transform=t)
    dl = DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)

    # Visualizes each batch in a 2x4 grid. Press ESC or q to exit.
    for batch in dl:
        visible_img = batch["images"]
        infrared_img = batch["ir_images"]
        boxes = batch["boxes"]
        classes = batch["classes"]
        batch_idx = batch["batch_idx"]

        batch_size = visible_img.shape[0]
        grid_h, grid_w = 2, 4
        img_h, img_w = visible_img.shape[-2:]
        canvas = np.zeros((grid_h * img_h, grid_w * img_w, 3), dtype=np.uint8)

        for i in range(batch_size):
            vis_img = from_tensor(visible_img[i], color_fmt="bgr")
            red_img = from_tensor(infrared_img[i], color_fmt="bgr")

            sample_mask = batch_idx == i
            sample_boxes = boxes[sample_mask]
            sample_classes = classes[sample_mask]

            for box, cls in zip(sample_boxes, sample_classes, strict=True):
                xyxy = box.int().tolist()
                rectangle(vis_img, xyxy, (0, 255, 0), thickness=2)
                rectangle(red_img, xyxy, (0, 255, 0), thickness=2)
                text_autoscale(
                    vis_img,
                    f"{cls.item()}",
                    xyxy[0:2],
                    (0, 255, 0),
                    move_base=True,
                )
                text_autoscale(
                    red_img,
                    f"{cls.item()}",
                    xyxy[0:2],
                    (0, 255, 0),
                    move_base=True,
                )

            col = i % grid_w
            x0, x1 = col * img_w, (col + 1) * img_w
            row = 0
            y0, y1 = row * img_h, (row + 1) * img_h
            canvas[y0:y1, x0:x1] = vis_img
            row = 1
            y0, y1 = row * img_h, (row + 1) * img_h
            canvas[y0:y1, x0:x1] = red_img

        cv2.imshow("KAIST8 Batch", canvas)
        key = cv2.waitKey(0)
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
