# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import Compose
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.instance import Instances


class _RTDETRToTvTensors:
    def __init__(self) -> None:
        from torchvision import tv_tensors

        self._tv_tensors = tv_tensors

    @staticmethod
    def _build_labels_tensor(cls: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(cls.reshape(-1), dtype=torch.int64) if len(cls) else torch.zeros((0,), dtype=torch.int64)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels.pop("img")
        instances = labels.pop("instances", None)
        cls = labels.pop("cls")

        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 3:
            image = Image.fromarray(img[..., ::-1].copy())  # BGR -> RGB
        else:
            image = Image.fromarray(img.copy())

        if instances is None or len(instances) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            instances.convert_bbox(format="xyxy")
            instances.denormalize(w, h)
            boxes_tensor = torch.as_tensor(instances.bboxes, dtype=torch.float32)

        labels["image"] = image
        labels["boxes"] = self._tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(h, w))
        labels["labels"] = self._build_labels_tensor(cls)
        return labels


class _RTDETRFromTvTensors:
    def __init__(self, scale_float: bool = False, normalize: bool = False) -> None:
        self.scale_float = bool(scale_float)
        self.normalize = bool(normalize)

    @staticmethod
    def _to_numpy_image_default(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            if img.ndim == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            img = np.asarray(image)
        if np.issubdtype(img.dtype, np.floating):
            if img.size and img.max() <= 1.0:
                img = img * 255.0
            img = img.round().astype(np.uint8)
        return img

    @staticmethod
    def _to_numpy_image_scaled(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            if img.ndim == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            img = np.asarray(image)
        img = img.astype(np.float32, copy=False)
        if img.size and img.max() > 1.0:
            img = img / 255.0
        return img

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
        return (img - mean) / std

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        image = labels.pop("image")
        boxes_t = labels.pop("boxes", None)
        labels_t = labels.pop("labels", None)

        if self.scale_float:
            img_np = self._to_numpy_image_scaled(image)
        else:
            img_np = self._to_numpy_image_default(image)
        if self.scale_float and self.normalize:
            img_np = self._normalize_image(img_np)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = img_np[..., ::-1]  # RGB -> BGR

        if boxes_t is None or boxes_t.numel() == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls_out = np.zeros((0, 1), dtype=np.int64)
        else:
            bboxes = boxes_t.to(torch.float32).cpu().numpy()
            if labels_t is None:
                cls_out = np.zeros((len(bboxes), 1), dtype=np.int64)
            else:
                cls_out = labels_t.to(torch.int64).view(-1, 1).cpu().numpy()

        labels["img"] = img_np
        labels["instances"] = Instances(bboxes=bboxes, bbox_format="xyxy", normalized=False)
        labels["cls"] = cls_out
        labels["img_scaled"] = self.scale_float
        # Keep a stable key set across mosaic/non-mosaic branches for downstream collate_fn.
        shape = tuple(img_np.shape[:2])
        labels.setdefault("im_file", "")
        labels.setdefault("ori_shape", shape)
        labels.setdefault("ratio_pad", (1.0, 1.0))
        labels["resized_shape"] = shape
        return labels


class _RTDETRRandomIoUCrop:
    def __init__(self, p: float = 1.0, **kwargs) -> None:
        import torchvision.transforms.v2 as T

        self.p = p
        self.transform = T.RandomIoUCrop(**kwargs)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(1) >= self.p:
            return labels
        return self.transform(labels)


def compute_deim_scheduled_prob(base_prob: float, epoch: int, stop_epoch: int, floor: float = 0.0) -> float:
    """Linearly decay an augmentation probability to floor by the no-aug stage boundary."""
    base_prob = float(base_prob)
    if base_prob <= 0.0 or stop_epoch <= 0:
        return 0.0
    if epoch >= stop_epoch:
        return 0.0
    return float(floor) + (base_prob - float(floor)) * max(0.0, 1.0 - (float(epoch) / float(stop_epoch)))


def resolve_deim_aug_scheduler(hyp: IterableSimpleNamespace | Any) -> str:
    """Resolve DEIM augmentation scheduler mode from config."""
    mode = str(hyp.deim_aug_scheduler).strip().lower()
    aliases = {
        "legacy": "legacy",
        "stage": "legacy",
        "staged": "legacy",
        "default": "legacy",
        "decay": "decay",
        "linear": "decay",
        "linear_decay": "decay",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported deim_aug_scheduler={mode!r}. Expected one of: legacy, decay.")
    return aliases[mode]


def rtdetr_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """Apply a series of image transformations for RT-DETR training."""
    del dataset, stretch  # Unused, kept for API compatibility with existing transform builders.
    import torchvision.transforms.v2 as T

    if not hasattr(hyp, "fliplr"):
        raise AttributeError("rtdetr_transforms requires 'fliplr' in hyp.")
    fliplr = float(hyp.fliplr)
    return Compose(
        [
            _RTDETRToTvTensors(),
            T.RandomPhotometricDistort(p=0.5),
            T.RandomZoomOut(fill=0),
            _RTDETRRandomIoUCrop(p=0.8),
            T.SanitizeBoundingBoxes(min_size=1),
            T.RandomHorizontalFlip(p=fliplr),
            T.Resize(size=[imgsz, imgsz]),
            T.SanitizeBoundingBoxes(min_size=1),
            _RTDETRFromTvTensors(),
        ]
    )  # transforms


class _RTDETRDEIMPolicy:
    """Epoch-aware DEIM transform policy with selectable staged or decay scheduling."""

    def __init__(
        self,
        dataset,
        imgsz: int,
        fliplr: float,
        policy_epochs: tuple[int, int, int],
        mosaic_prob: float,
        scheduler_mode: str = "legacy",
        normalize_input: bool = False,
        mosaic_use_cache: bool = False,
        mosaic_max_cached_images: int = 50,
        mosaic_random_pop: bool = True,
    ) -> None:
        import torchvision.transforms.v2 as T

        self.to_tv = _RTDETRToTvTensors()
        # DEIM Mosaic op itself runs with probability 1.0; branch routing is handled externally via mosaic_prob.
        self.mosaic = _RTDETRDEIMMosaic(
            dataset,
            imgsz=imgsz,
            p=1.0,
            use_cache=mosaic_use_cache,
            max_cached_images=mosaic_max_cached_images,
            random_pop=mosaic_random_pop,
        )
        self.photometric = T.RandomPhotometricDistort(p=0.5)
        self.zoomout = T.RandomZoomOut(fill=0)
        self.ioucrop = _RTDETRRandomIoUCrop(p=0.8)
        self.sanitize1 = T.SanitizeBoundingBoxes(min_size=1)
        self.flip = T.RandomHorizontalFlip(p=fliplr)
        self.resize = T.Resize(size=[imgsz, imgsz])
        self.sanitize2 = T.SanitizeBoundingBoxes(min_size=1)
        self.from_tv = _RTDETRFromTvTensors(scale_float=True, normalize=normalize_input)

        self.policy_epochs = policy_epochs
        self.scheduler_mode = str(scheduler_mode)
        self.base_mosaic_prob = float(mosaic_prob)
        self.mosaic_prob = self.base_mosaic_prob
        self.epoch = 0
        self.post_transforms = []

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch (0-based) for stage scheduling."""
        self.epoch = epoch
        _, _, stop = self.policy_epochs
        if self.scheduler_mode == "decay":
            self.mosaic_prob = compute_deim_scheduled_prob(self.base_mosaic_prob, epoch, stop)
        else:
            self.mosaic_prob = self.base_mosaic_prob

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        start, mid, stop = self.policy_epochs
        cur_epoch = self.epoch

        if start <= cur_epoch < mid:
            with_mosaic = random.random() <= self.mosaic_prob
            labels = self.mosaic(labels) if with_mosaic else self.to_tv(labels)
            labels = self.photometric(labels)
            if not with_mosaic:
                labels = self.zoomout(labels)
                labels = self.ioucrop(labels)
        elif mid <= cur_epoch < stop:
            labels = self.to_tv(labels)
            labels = self.photometric(labels)
            labels = self.zoomout(labels)
            labels = self.ioucrop(labels)
        else:
            labels = self.to_tv(labels)

        # Always-on ops
        labels = self.sanitize1(labels)
        labels = self.flip(labels)
        labels = self.resize(labels)
        labels = self.sanitize2(labels)
        labels = self.from_tv(labels)
        for transform in self.post_transforms:
            labels = transform(labels)
        return labels

    def append(self, transform) -> None:
        """Append post-transform ops (e.g., Format) for API compatibility with Compose-like callers."""
        self.post_transforms.append(transform)


class _RTDETRDEIMMosaic:
    """DEIM-style Mosaic that keeps data in torchvision tv_tensor format."""

    def __init__(
        self,
        dataset,
        imgsz: int = 640,
        p: float = 1.0,
        use_cache: bool = False,
        max_cached_images: int = 50,
        random_pop: bool = True,
    ) -> None:
        import torchvision.transforms.v2 as T
        from torchvision import tv_tensors

        self.dataset = dataset
        self.half_size = imgsz // 2
        self.p = p
        self.use_cache = bool(use_cache)
        self.max_cached_images = max(1, int(max_cached_images))
        self.random_pop = bool(random_pop)
        self.mosaic_cache: list[dict[str, Any]] = []
        self._tv_tensors = tv_tensors
        # Single int → torchvision resizes the shorter edge to half_size, preserving aspect ratio.
        # This matches DEIM's `T.Resize(size=output_size)` behaviour exactly.
        self._resize = T.Resize(size=self.half_size)
        self._affine = T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=0)

    def _convert_to_pil(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels.pop("img")
        instances = labels.pop("instances", None)
        cls = labels.pop("cls")

        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 3:
            image = Image.fromarray(img[..., ::-1].copy())  # BGR -> RGB
        else:
            image = Image.fromarray(img.copy())

        if instances is None or len(instances) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            instances.convert_bbox(format="xyxy")
            if instances.segments is None:
                instances.segments = np.zeros((0, 0, 2), dtype=np.float32)
            instances.denormalize(w, h)
            boxes_tensor = torch.as_tensor(instances.bboxes, dtype=torch.float32)

        cls_tensor = (
            torch.as_tensor(cls.reshape(-1), dtype=torch.int64) if len(cls) else torch.zeros((0,), dtype=torch.int64)
        )

        labels["image"] = image
        labels["boxes"] = self._tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(h, w))
        labels["labels"] = cls_tensor
        return labels

    def _clone_mosaic_labels(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Clone mosaic sample dict for cache re-use without shared mutable state."""
        boxes = labels["boxes"]
        return {
            "image": labels["image"].copy(),
            "boxes": self._tv_tensors.BoundingBoxes(
                boxes.clone().to(torch.float32),
                format="XYXY",
                canvas_size=tuple(boxes.canvas_size),
            ),
            "labels": labels["labels"].clone(),
        }

    def _mosaic4(self, labels_list: list[dict[str, Any]]) -> dict[str, Any]:
        # Derive canvas dimensions from the actual tile sizes after aspect-preserving resize.
        # PIL .width / .height give (W, H); torchvision canvas_size convention is (H, W).
        max_height = max(lbl["image"].height for lbl in labels_list)
        max_width = max(lbl["image"].width for lbl in labels_list)
        canvas_w = max_width * 2   # PIL width
        canvas_h = max_height * 2  # PIL height

        # PIL paste: tiles placed at quadrant corners; unfilled regions stay 0 (black).
        # Matches DEIM's `Image.new(..., color=0)` + `image.paste(img, placement_offsets[i])`.
        mode = labels_list[0]["image"].mode
        merged_image = Image.new(mode, (canvas_w, canvas_h), color=0)
        # (x_offset, y_offset) in PIL/pixel space  ←→  DEIM's [[0,0],[max_width,0],[0,max_height],[max_width,max_height]]
        offsets = [(0, 0), (max_width, 0), (0, max_height), (max_width, max_height)]

        all_boxes, all_cls = [], []
        for lbl, (x_off, y_off) in zip(labels_list, offsets):
            merged_image.paste(lbl["image"], (x_off, y_off))
            boxes = lbl["boxes"]
            if len(boxes):
                # XYXY shift: add [x_off, y_off, x_off, y_off] — matches DEIM's offset tensor
                offset_t = boxes.new_tensor([x_off, y_off, x_off, y_off], dtype=torch.float32)
                all_boxes.append(boxes.clone().to(torch.float32) + offset_t)
                all_cls.append(lbl["labels"])

        if all_boxes:
            final_boxes = torch.cat(all_boxes, dim=0)
            final_cls = torch.cat(all_cls, dim=0)
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_cls = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": merged_image,
            # canvas_size is (H, W) per torchvision convention
            "boxes": self._tv_tensors.BoundingBoxes(final_boxes, format="XYXY", canvas_size=(canvas_h, canvas_w)),
            "labels": final_cls,
        }

    def _load_samples_from_dataset(self, labels: dict[str, Any]) -> list[dict[str, Any]]:
        labels["image"], labels["boxes"] = self._resize(labels["image"], labels["boxes"])
        sample_indices = random.choices(range(len(self.dataset)), k=3)
        all_labels = [labels]
        for idx in sample_indices:
            other = self.dataset.get_image_and_label(idx)
            other = self._convert_to_pil(other)
            other["image"], other["boxes"] = self._resize(other["image"], other["boxes"])
            all_labels.append(other)
        return all_labels

    def _load_samples_from_cache(self, labels: dict[str, Any]) -> list[dict[str, Any]]:
        labels["image"], labels["boxes"] = self._resize(labels["image"], labels["boxes"])
        self.mosaic_cache.append(self._clone_mosaic_labels(labels))
        if len(self.mosaic_cache) > self.max_cached_images:
            if self.random_pop and len(self.mosaic_cache) > 1:
                pop_idx = random.randint(0, len(self.mosaic_cache) - 2)  # keep the latest sample
            else:
                pop_idx = 0
            self.mosaic_cache.pop(pop_idx)

        sample_indices = random.choices(range(len(self.mosaic_cache)), k=3)
        all_labels = [self._clone_mosaic_labels(labels)]
        for idx in sample_indices:
            all_labels.append(self._clone_mosaic_labels(self.mosaic_cache[idx]))
        return all_labels

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels = self._convert_to_pil(labels)
        if random.random() > self.p:
            return labels

        all_labels = self._load_samples_from_cache(labels) if self.use_cache else self._load_samples_from_dataset(labels)

        mosaic_labels = self._mosaic4(all_labels)
        mosaic_labels["image"], mosaic_labels["boxes"] = self._affine(mosaic_labels["image"], mosaic_labels["boxes"])
        return mosaic_labels


def compute_policy_epochs(hyp: IterableSimpleNamespace) -> tuple[int, int, int]:
    """Compute DEIM policy boundaries.

    Supports optional DEIM-style schedule key:
      - flat_epoch: explicit stage-2 end / stage-3 start epoch
      - no_aug_epoch: explicit no-augmentation tail length at the end of training
    """
    if not hasattr(hyp, "epochs"):
        raise AttributeError("compute_policy_epochs requires 'epochs' in hyp.")

    epochs = max(1, int(hyp.epochs))

    explicit_no_aug = getattr(hyp, "no_aug_epoch", None) is not None
    if explicit_no_aug:
        no_aug_epoch = int(hyp.no_aug_epoch)
        if no_aug_epoch < 0:
            raise ValueError(f"compute_policy_epochs got invalid no_aug_epoch={no_aug_epoch}. Expected >= 0.")
    else:
        # Mimic DEIM RT schedules from total epochs only:
        #   60 -> 2 no-aug epochs, 120 -> 3 no-aug epochs.
        # Keep classic 50-epoch behavior with no final no-aug tail.
        if epochs >= 100:
            no_aug_epoch = 3
        elif epochs >= 60:
            no_aug_epoch = 2
        else:
            no_aug_epoch = 0

    stop = epochs - no_aug_epoch
    if stop < 0:
        raise ValueError(f"compute_policy_epochs got invalid no_aug_epoch={no_aug_epoch} for epochs={epochs}.")

    # DEIM-style policy epochs are derived on the active-augmentation span when no_aug is explicit.
    effective_epochs = stop if explicit_no_aug else epochs
    start = min(4, max(0, effective_epochs - 1))

    flat_epoch = hyp.flat_epoch
    if flat_epoch is None:
        mid = min(stop, start + effective_epochs // 2)
    else:
        mid = int(flat_epoch)
    if not (0 <= start <= mid <= stop <= epochs):
        raise ValueError(
            f"compute_policy_epochs produced invalid boundaries: start={start}, mid={mid}, stop={stop}, epochs={epochs}."
        )
    return start, mid, stop


def rtdetr_deim_transforms(
    dataset,
    imgsz: int,
    hyp: IterableSimpleNamespace,
    policy_epochs: tuple[int, int, int],
    mosaic_prob: float,
    stretch: bool = False,
):
    """Build epoch-aware DEIM transforms for RT-DETR variants."""
    del stretch  # Unused, kept for API compatibility with existing transform builders.
    if not hasattr(hyp, "fliplr"):
        raise AttributeError("rtdetr_deim_transforms requires 'fliplr' in hyp.")
    fliplr = float(hyp.fliplr)
    normalize_input = bool(hyp.rtdetr_input_normalize)
    mosaic_use_cache = bool(hyp.mosaic_use_cache)
    mosaic_max_cached_images = 50
    mosaic_random_pop = True
    scheduler_mode = resolve_deim_aug_scheduler(hyp)
    return _RTDETRDEIMPolicy(
        dataset=dataset,
        imgsz=imgsz,
        fliplr=fliplr,
        policy_epochs=policy_epochs,
        mosaic_prob=float(mosaic_prob),
        scheduler_mode=scheduler_mode,
        normalize_input=normalize_input,
        mosaic_use_cache=mosaic_use_cache,
        mosaic_max_cached_images=mosaic_max_cached_images,
        mosaic_random_pop=mosaic_random_pop,
    )
