import argparse
import importlib
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torchmetrics as M
from torchvision.transforms import v2 as T


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to config file",
        metavar="FILE",
    )
    args = parser.parse_args()
    return main(args)


def main(args: argparse.Namespace) -> None:
    with args.config.open("rb") as f:
        config = tomllib.load(f)

    model_type = dynamic_import(config["model"])
    model_args = parse_kwargs(config["model"], _locals=locals())
    model = model_type(**model_args)

    optimizer_type = dynamic_import(config["optimizer"])
    optimizer_args = parse_kwargs(config["optimizer"], _locals=locals())
    optimizer = optimizer_type(model.parameters(), **optimizer_args)

    loss_fn_type = dynamic_import(config["loss_fn"])
    loss_fn_args = parse_kwargs(config["loss_fn"], _locals=locals())
    loss_fn = loss_fn_type(**loss_fn_args)

    metrics = config.get("metrics", {})
    for k, v in metrics.items():
        metric_type = dynamic_import(v)
        metric_args = parse_kwargs(v, _locals=locals())
        metrics[k] = metric_type(**metric_args)

    callbacks = config["trainer"].get("callbacks", [])
    for i, cb in enumerate(callbacks):
        callback_type = dynamic_import(cb)
        callback_args = parse_kwargs(cb, _locals=locals())
        callbacks[i] = callback_type(**callback_args)

    for k, v in config["trainer"].items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                if not isinstance(item, dict):
                    break
                item_type = dynamic_import(item)
                item_args = parse_kwargs(item, _locals=locals())
                v[i] = item_type(**item_args)
        elif isinstance(v, dict):
            item_type = dynamic_import(v)
            item_args = parse_kwargs(v, _locals=locals())
            config["trainer"][k] = item_type(**item_args)

    wrapper = LitModule(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics,
    )
    dataset = LitDataModule(config["dataset"])
    trainer = L.Trainer(**config["trainer"])

    trainer.fit(model=wrapper, datamodule=dataset)


def dynamic_import(ctx: dict[str, Any]) -> Any:
    """根據提供的模組資訊動態引入符號。

    需要的模組資訊以字典形式提供：

    ```python
    {
        "module": {
            "path": "python.module.path",
            "name": "ClassName",
        }
    }
    ```

    Args:
        ctx (dict[str, Any]): 包含模組路徑、名稱的字典。

    Returns:
        out (Any): 引入的符號。
    """
    module_info = ctx["module"]
    module_path = module_info["path"]
    module_name = module_info["name"]
    py_module = importlib.import_module(module_path)
    return getattr(py_module, module_name)


def parse_kwargs(ctx: dict[str, Any], /, _globals=None, _locals=None) -> dict[str, Any]:
    """解析字典中需要動態展開為 Python 物件的字串。

    會將以 "$" 開頭的字串值視為 Python 表達式，並以 eval(...) 的結果替換原字串值。

    ```python
    {
        "module": {
            "args": {
                "name1": "value1",
                "name2": "$expression",
            }
        }
    }
    ```

    Args:
        ctx (dict[str, Any]): 包含引數的字典。

    Returns:
        out (dict[str, Any]): 解析後的引數字典。
    """
    args_dict = ctx["module"].get("args", {})
    for k, v in args_dict.items():
        if isinstance(v, str) and v.startswith("$"):
            args_dict[k] = eval(v[1:], globals=_globals, locals=_locals)
    return args_dict


class LitModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict[str, M.Metric],
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        predictions = self.model(batch["images"])
        loss = self.loss_fn(predictions, batch)
        loss_items = loss.detach()
        return {"loss": loss.sum(), "loss_items": loss_items}

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        predictions = self.model(batch["images"])
        return {"loss": None, "predictions": predictions}

    def on_train_epoch_end(self) -> None:
        # Resets all metrics at the end of each training epoch.
        for m in self.metrics.values():
            m.reset()

    def configure_optimizers(self):
        return self.optimizer


class LossCallback(L.Callback):
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: LitModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert isinstance(outputs, Mapping) and "loss_items" in outputs, (
            "Expected outputs to be a mapping containing 'loss_items'."
        )
        loss_items = outputs["loss_items"]
        pl_module.log_dict(
            {
                "total_loss": loss_items.sum(),
                "box_loss": loss_items[0],
                "cls_loss": loss_items[1],
                "dfl_loss": loss_items[2],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class MAPCallback(L.Callback):
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: LitModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(outputs, Mapping) and "predictions" in outputs, (
            "Expected outputs to be a mapping containing 'predictions'."
        )
        # Formats predictions for metric computation.
        preds = []
        for pred in outputs["predictions"]:
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
        _, counts = batch["batch_idx"].unique_consecutive(return_counts=True)
        counts = counts.tolist()
        for i in range(len(counts)):
            # This datasets has only 1 box in each image, so we can directly take them.
            num_valid = counts[i]
            targets.append(
                {
                    "boxes": batch["boxes"][i, 0:num_valid],
                    "labels": batch["classes"][i, 0:num_valid].long(),
                }
            )

        # Updates metrics.
        pl_module.metrics["map"].update(preds, targets)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: LitModule) -> None:
        result = pl_module.metrics["map"].compute()
        pl_module.log_dict(
            {
                "mAP50-95": result["map"],
                "mAP50": result["map_50"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasets: dict[str, Any],
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = datasets.get("batch_size", 8)
        self.num_workers = datasets.get("num_workers", 0)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._setup_split(self.datasets["train"])
            self._setup_split(self.datasets["validate"])
        else:  # stage in ["validate", "test", "predict"]
            self._setup_split(self.datasets[stage])

    def _setup_split(self, split: dict[str, Any]) -> None:
        if modules := split.get("transforms", []):
            # Initialize transform instances.
            transforms = [dynamic_import(m)(**parse_kwargs(m)) for m in modules]
            split["module"]["args"]["transform"] = T.Compose(transforms)
        # Initialize dataset instance.
        split["instance"] = dynamic_import(split)(**parse_kwargs(split))

    def train_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.datasets["train"]["instance"],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.datasets["train"]["instance"].collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.datasets["validate"]["instance"],
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.datasets["validate"]["instance"].collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.datasets["test"]["instance"],
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.datasets["test"]["instance"].collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(
            self.datasets["predict"]["instance"],
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.datasets["predict"]["instance"].collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
