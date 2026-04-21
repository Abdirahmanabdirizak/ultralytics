# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import torch.distributed as dist
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, RANK
from ultralytics.utils.afss import AFSSScheduler
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)
        if self.args.afss:
            self.add_callback("on_train_epoch_start", self._afss_on_epoch_start)
            self.add_callback("on_train_epoch_end", self._afss_on_epoch_end)
            self.add_callback("on_model_save", self._afss_save_state)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train", active_indices=None
    ):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.
            active_indices (list[int], optional): Active image indices for AFSS sampling.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        if active_indices is not None:
            dataset.active_indices = active_indices
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens, cls_pw > 1 amplifies).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw  # apply power directly
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def _unwrap_dataset(self, dataset):
        """Unwrap a dataset from any dataloader wrapper layers."""
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def _afss_on_epoch_start(self):
        """AFSS callback: sample active indices at the start of each epoch after warmup."""
        if not hasattr(self, "afss_scheduler"):
            # Lazy init on first epoch
            dataset = self._unwrap_dataset(self.train_loader.dataset)

            self.afss_scheduler = AFSSScheduler(
                len(dataset), warmup_epochs=self.args.warmup_epochs, seed=self.args.seed
            )
            self.afss_tau = self.afss_scheduler.tau
            self.afss_current_indices = list(range(len(dataset)))

            # Resume: restore scheduler state if available
            afss_path = self.wdir / "afss_state.pt"
            if afss_path.exists():
                state = torch.load(afss_path, weights_only=False)
                self.afss_scheduler.state = state

        epoch = self.epoch
        if epoch < self.afss_tau:
            return

        selected_indices = self.afss_scheduler.sample_indices(epoch)

        # DDP broadcast
        if self.world_size > 1:
            if RANK == 0:
                broadcast_list = [selected_indices]
            else:
                broadcast_list = [None]
            dist.broadcast_object_list(broadcast_list, src=0)
            selected_indices = broadcast_list[0]

        if self.world_size > 1:
            # Rebuild loader for DDP so DistributedSampler sees new length
            batch_size = self.batch_size // self.world_size
            self.train_loader = self.get_dataloader(
                self.data["train"],
                batch_size=batch_size,
                rank=LOCAL_RANK,
                mode="train",
                active_indices=selected_indices,
            )

            new_dataset = self._unwrap_dataset(self.train_loader.dataset)

            if self.args.close_mosaic and epoch >= (self.epochs - self.args.close_mosaic):
                new_dataset.close_mosaic(hyp=copy(self.args))
        else:
            dataset = self._unwrap_dataset(self.train_loader.dataset)
            dataset.active_indices = selected_indices
            self.train_loader.reset()

        self.afss_current_indices = selected_indices
        self.nb = len(self.train_loader)
        # Adjust last_opt_step so optimizer stepping continues correctly when nb changes
        self.last_opt_step = epoch * self.nb - self.accumulate
        LOGGER.info(f"AFSS epoch {epoch}: training on {len(selected_indices)}/{self.afss_scheduler.num_images} images")

    def _afss_on_epoch_end(self):
        """AFSS callback: update last seen and refresh metrics at the end of each epoch."""
        if not hasattr(self, "afss_scheduler"):
            return
        epoch = self.epoch
        self.afss_scheduler.update_last_seen(self.afss_current_indices, epoch)
        if epoch >= self.afss_tau and (epoch - self.afss_tau) % 5 == 0:
            self._afss_refresh_metrics()

    def _afss_refresh_metrics(self):
        """Run validation on the training set to refresh per-image precision/recall for AFSS."""
        from pathlib import Path

        batch_size = self.batch_size // max(self.world_size, 1)
        train_eval_loader = self.get_dataloader(self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="val")

        validator = self.get_validator().__class__(
            train_eval_loader,
            save_dir=self.save_dir / "afss_train_eval",
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator(self)

        if RANK in {-1, 0}:
            image_metrics = validator.metrics.box.image_metrics
            dataset = self._unwrap_dataset(self.train_loader.dataset)
            filename_to_idx = {Path(f).name: i for i, f in enumerate(dataset.im_files)}
            self.afss_scheduler.update_metrics(image_metrics, filename_to_idx)
            LOGGER.info(f"AFSS: refreshed metrics for {len(image_metrics)} images")

        if self.world_size > 1:
            state_list = [self.afss_scheduler.state if RANK == 0 else None]
            dist.broadcast_object_list(state_list, src=0)
            self.afss_scheduler.state = state_list[0]

    def _afss_save_state(self):
        """Save AFSS scheduler state to a sidecar checkpoint file."""
        if hasattr(self, "afss_scheduler") and RANK in {-1, 0}:
            torch.save(self.afss_scheduler.state, self.wdir / "afss_state.pt")

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)
