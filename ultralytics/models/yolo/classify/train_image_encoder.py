# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Trainer for encoder distillation pretraining from frozen vision foundation models.

Distill one or more teachers (EUPE, DINOv3, SAM3, SigLIP2) into a YOLO backbone using online
teacher forward each step. Supports single-teacher (EUPE Section 4: proxy -> student) and
multi-teacher (EUPE Section 3: multiple teachers -> proxy, Eq.6: L = sum_i L_i).

Dataset support: WebDataset tar shards (DataComp-12M) and image folders (COCO, ImageNet).
"""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ultralytics.data.augment import classify_augmentations, classify_transforms
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.image_encoder import ImageEncoderModel
from ultralytics.nn.teacher_model import TEACHER_REGISTRY, build_teacher_model, safe_key
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

# ImageNet normalization (used by EUPE, DINOv3, SigLIP2, SAM3 -- standard for ViT models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _WebDatasetLoader(DataLoader):
    """DataLoader for WebDataset shards, compatible with ultralytics trainer.

    Wraps a wds.DataPipeline inside a standard DataLoader by converting it to an IterableDataset. Provides .reset(),
    .dataset (with __len__), .num_workers, .sampler -- all required by engine/trainer.py (lines 283, 370, 379, 403).

    Args:
        pipeline: WebDataset pipeline (wds.DataPipeline).
        num_samples (int): Estimated total samples in dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
    """

    def __init__(self, pipeline, num_samples, batch_size, num_workers, drop_last=False):
        """Initialize with WebDataset pipeline wrapped as IterableDataset."""

        class _IterDS(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter(pipeline)

            def __len__(self):
                return num_samples

        super().__init__(
            _IterDS(), batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=drop_last
        )

    def reset(self):
        """No-op: WebDataset workers don't need reset (no mosaic state)."""


class _ImageOnlyDataset(Dataset):
    """Load all images from a directory for label-free distillation.

    Walk a directory recursively for image files. No class labels needed -- teacher provides the supervision signal.
    """

    def __init__(self, root, transform):
        """Initialize with directory path and transform.

        Args:
            root (str | Path): Directory containing images (flat or nested).
            transform: Callable that takes PIL image and returns a tensor.
        """
        self.samples = sorted(p for p in Path(root).rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
        self.transform = transform
        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}")
        LOGGER.info(f"ImageOnlyDataset: {len(self.samples)} images from {root}")

    def __len__(self):
        """Return number of images."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Load image and apply transform."""
        return self.transform(Image.open(self.samples[index]).convert("RGB"))


class ImageEncoderTrainer(ClassificationTrainer):
    """Trainer for single or multi-teacher encoder distillation pretraining.

    Single-teacher: EUPE Section 4 (proxy -> student). Multi-teacher: EUPE Section 3 / Eq.6 (sum over teachers). All
    teachers run online each step (frozen, no pre-computed embeddings) following AM-RADIO/EUPE convention.

    Attributes:
        teacher_names (list[str]): Teacher variants (e.g. ['eupe:vitb16'] or ['eupe:vitb16', 'dinov3:vitl16']).
        teachers (dict): Loaded frozen teacher models keyed by safe name (e.g. 'eupe_vitb16').
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize ImageEncoderTrainer.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary.
            overrides (dict[str, Any], optional): Parameter overrides. Supports 'teacher_name' (single, e.g.
            'eupe: vitb16') or 'teacher_names' (multi, '+' separated, e.g. 'eupe:vitb16+dinov3:vitl16').
            _callbacks (dict, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        overrides.setdefault("close_mosaic", 0)  # no mosaic in distillation, avoids .reset() call
        # Support both 'teacher_name' (single) and 'teacher_names' (multi, '+' separated)
        raw = overrides.pop("teacher_names", overrides.pop("teacher_name", "eupe:vitb16"))
        self.teacher_names = raw.split("+") if isinstance(raw, str) else raw
        self._safe_keys = [safe_key(n) for n in self.teacher_names]
        self.teachers = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self):
        """Build minimal data dict for distillation (no check_cls_dataset needed).

        Auto-detect layout: shards/*.tar (WebDataset), images/train2017 (COCO), train/ (ImageNet), or flat folder.

        Returns:
            (dict): Data dict with 'train', 'val', 'nc', 'names', 'channels' keys.
        """
        data_path = Path(self.args.data)
        if (data_path / "shards").is_dir():
            train_path = val_path = str(data_path)
        elif (data_path / "images" / "train2017").is_dir():
            train_path, val_path = str(data_path / "images" / "train2017"), str(data_path / "images" / "val2017")
        elif (data_path / "train").is_dir():
            train_path, val_path = str(data_path / "train"), str(data_path / "val")
        else:
            train_path = val_path = str(data_path)
        return {
            "train": train_path,
            "val": val_path,
            "nc": 1000,
            "names": {i: str(i) for i in range(1000)},
            "channels": 3,
        }

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return ImageEncoderModel with per-teacher adaptor heads.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (ImageEncoderModel): Model with per-teacher adaptor heads.
        """
        self._load_teachers()
        # Build teacher config dict for the model
        teachers_cfg = {}
        for name in self.teacher_names:
            reg = TEACHER_REGISTRY[name]
            teachers_cfg[name] = {
                "embed_dim": reg["embed_dim"],
                "num_patches": reg["num_patches"],
                "token_types": reg["token_types"],
            }
        model = ImageEncoderModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1, teachers=teachers_cfg
        )
        if weights:
            model.load(weights)
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        model.model[-1].linear.requires_grad_(False)
        return model

    def _load_teachers(self):
        """Load and cache all frozen teacher models."""
        for name, sk in zip(self.teacher_names, self._safe_keys):
            if sk not in self.teachers:
                LOGGER.info(f"Loading teacher '{name}'...")
                self.teachers[sk] = build_teacher_model(name, self.device)
                n = sum(p.numel() for p in self.teachers[sk].parameters()) / 1e6
                LOGGER.info(f"  {name}: {n:.1f}M params, embed_dim={self.teachers[sk].embed_dim}")

    def _build_transforms(self, mode):
        """Build shared transform at teacher resolution (256) with ImageNet normalization.

        Same augmented image goes to both teacher and student, resized to student resolution in
        preprocess_batch (EUPE Stage 2 / DUNE / AM-RADIO convention).
        """
        if mode == "train":
            return classify_augmentations(
                size=256,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                hflip=self.args.fliplr,
                vflip=self.args.flipud,
                erasing=self.args.erasing,
                auto_augment=self.args.auto_augment,
                hsv_h=self.args.hsv_h,
                hsv_s=self.args.hsv_s,
                hsv_v=self.args.hsv_v,
                interpolation="BICUBIC",
            )
        return classify_transforms(size=256, mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="BICUBIC")

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build dataset from WebDataset shards or image folder.

        Args:
            img_path (str): Path to dataset (shards dir or image folder).
            mode (str, optional): Dataset mode.
            batch (Any, optional): Unused.

        Returns:
            Dataset yielding (student_tensor, teacher_tensor) tuples.
        """
        tf = self._build_transforms(mode)

        shards = sorted(str(p) for p in Path(img_path).glob("shards/*.tar"))
        if shards:
            import webdataset as wds

            return (
                wds.WebDataset(shards, shardshuffle=mode == "train")
                .shuffle(1000 if mode == "train" else 0)
                .decode("pil", handler=wds.warn_and_continue)
                .to_tuple("jpg", handler=wds.warn_and_continue)
                .map(lambda sample: tf(sample[0]))
                .with_epoch(len(shards) * 10000)
            )
        return _ImageOnlyDataset(img_path, tf)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return DataLoader for WebDataset or image folder.

        Args:
            dataset_path (str): Path to dataset root.
            batch_size (int, optional): Batch size.
            rank (int, optional): Process rank for DDP.
            mode (str, optional): 'train' or 'val'.

        Returns:
            (DataLoader): DataLoader yielding (student_imgs, teacher_imgs) batches.
        """
        dataset = self.build_dataset(dataset_path, mode)
        if isinstance(dataset, torch.utils.data.IterableDataset):
            # WebDataset pipeline -- wrap for DataLoader compatibility
            num_samples = 0
            for f in sorted(Path(dataset_path).glob("shards/*_stats.json")):
                with open(f) as fh:
                    num_samples += json.load(fh)["successes"]
            if not num_samples:
                num_samples = len(list(Path(dataset_path).glob("shards/*.tar"))) * 6000
                LOGGER.warning(f"No stats files found, estimating {num_samples} samples from shard count")
            return _WebDatasetLoader(dataset, num_samples, batch_size, self.args.workers, drop_last=mode == "train")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def preprocess_batch(self, batch):
        """Move images to device, resize for student, run all teachers.

        Single augmented image at 256x256 (teacher resolution) is resized to student resolution
        via F.interpolate. Follows DUNE convention (dune/teachers/forward.py:30).

        Args:
            batch (torch.Tensor): Images at teacher resolution (B, 3, 256, 256).

        Returns:
            (dict): Batch with 'img', 'cls', per-teacher entries, and '_teacher_keys'.
        """
        imgs = batch.to(self.device, non_blocking=True)
        # Resize to student resolution if different (DUNE: dune/teachers/forward.py:30)
        student_imgs = (
            torch.nn.functional.interpolate(imgs, size=self.args.imgsz, mode="bilinear", antialias=True)
            if self.args.imgsz != 256
            else imgs
        )

        result = {
            "img": student_imgs,
            "cls": torch.zeros(imgs.shape[0], dtype=torch.long, device=self.device),
            "_teacher_keys": self._safe_keys,
        }

        for sk in self._safe_keys:
            out = self.teachers[sk].encode(imgs)
            result[sk] = {"cls": out.cls, "patches": out.patches}

        return result

    def get_validator(self):
        """Return ImageEncoderValidator for loss-only validation."""
        from ultralytics.models.yolo.classify.val_image_encoder import ImageEncoderValidator

        # Per-teacher loss names: {safe_name}/cls_cos, {safe_name}/patch_cos, {safe_name}/patch_l1
        self.loss_names = []
        for sk in self._safe_keys:
            self.loss_names.extend([f"{sk}/cls_cos", f"{sk}/patch_cos", f"{sk}/patch_l1"])

        validator = ImageEncoderValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        validator.teachers = self.teachers
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return labeled loss items for WandB logging.

        Args:
            loss_items (torch.Tensor, optional): Loss items tensor.
            prefix (str, optional): Prefix for loss names.

        Returns:
            (dict | list): Labeled loss dict or list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))

    def plot_training_samples(self, batch, ni):
        """Skip training sample plotting for distillation (no class labels)."""
        pass

    def final_eval(self):
        """Skip final eval with best.pt (no classification accuracy for distillation)."""
        pass

    def plot_metrics(self):
        """Skip metric plotting for distillation (non-standard loss columns; use WandB)."""
