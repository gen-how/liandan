import argparse
import tomllib
from pathlib import Path
from typing import Any

# Parses CLI arguments brfore importing other heavy modules.
PARSER = argparse.ArgumentParser()
PARSER.add_argument("-c", "--config", type=Path, help="path to config file", required=True, metavar="FILE")  # fmt: skip # noqa: E501
ARGS = PARSER.parse_args()

import albumentations as A
import lightning as L
import torch
from torchmetrics.detection import MeanAveragePrecision

from liandan.data import BananaDetection
from liandan.losses.detection import YOLOv8DetectionLoss
from liandan.models import YOLOv8


class LitModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: dict[str, Any],
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self._entry: dict[str, torch.Tensor] = {}

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        predictions = self.model(batch["images"])
        loss = self.loss_fn(predictions, batch)

        batch_size = batch["images"].shape[0]
        loss_items = loss.detach() / batch_size
        self._entry["train/total_loss"] = loss_items.sum()
        self._entry["train/box_loss"] = loss_items[0]
        self._entry["train/cls_loss"] = loss_items[1]
        self._entry["train/dfl_loss"] = loss_items[2]

        # Returns values via dict so that the data can be accessed by subsequent hooks.
        return {"loss": loss.sum()}

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # Formats predictions for metric computation.
        predictions = self.model(batch["images"])
        preds = []
        for pred in predictions:
            mask = pred[:, 4] >= 0.05  # score_threshold, may be set in config?
            filtered_pred = pred[mask]
            preds.append(
                {
                    "boxes": filtered_pred[:, :4],
                    "scores": filtered_pred[:, 4],
                    "labels": filtered_pred[:, 5].long(),
                }
            )

        # Formats targets from batch for metric computation.
        targets = []
        for i in range(batch["images"].shape[0]):
            mask = batch["batch_idx"] == i
            targets.append(
                {
                    "boxes": batch["boxes"][mask],
                    "labels": batch["classes"][mask].squeeze(-1),
                }
            )

        # Updates metrics.
        self.metrics["map"].update(preds, targets)
        return None

    def on_train_epoch_end(self) -> None:
        result = self.metrics["map"].compute()
        self._entry["val/mAP@50-95"] = result["map"]
        self._entry["val/mAP@50"] = result["map_50"]
        self.log_dict(self._entry, on_step=False, on_epoch=True, prog_bar=True)
        # Resets all metrics at the end of each training epoch.
        for m in self.metrics.values():
            m.reset()


def main() -> None:
    # Loads config from file for default hyperparameters.
    with ARGS.config.open("rb") as f:
        cfg = tomllib.load(f)

    # Hyperparameters can be overridden by CLI arguments.
    # TODO: implement CLI overrides config.

    # Initializes components.
    model = YOLOv8(
        cfg["model"]["version"],
        cfg["model"]["num_classes"],
        cfg["model"]["img_size"],
        cfg["model"]["reg_max"],
    )
    loss_fn = YOLOv8DetectionLoss(model.strides, model.num_classes, model.reg_max)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        cfg["trainer"]["max_epochs"],
        cfg["optimizer"]["lr"] * 0.01,
    )

    metrics = {"map": MeanAveragePrecision(box_format="xyxy", iou_type="bbox")}

    img_h, img_w = cfg["model"]["img_size"]
    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=img_h, width=img_w),
            A.ToFloat(),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["classes"]),
    )
    train_dataset = BananaDetection(
        cfg["dataset"]["root"],
        "train",
        train_transforms,
        cfg["dataset"]["download"],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=cfg["dataset"]["num_workers"],
        persistent_workers=cfg["dataset"]["num_workers"] > 0,
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.ToFloat(),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),
            A.ToTensorV2(),
        ],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["classes"]),
    )
    val_dataset = BananaDetection(
        cfg["dataset"]["root"],
        "valid",
        val_transforms,
        cfg["dataset"]["download"],
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=cfg["dataset"]["num_workers"],
        persistent_workers=cfg["dataset"]["num_workers"] > 0,
    )

    wrapper = LitModule(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
    )

    trainer = L.Trainer(**cfg["trainer"])
    trainer.fit(
        model=wrapper,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
