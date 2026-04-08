# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Depth estimation validator for YOLO models."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER


class DepthValidator(DetectionValidator):
    """Validator for YOLO depth estimation models.

    Extends DetectionValidator with depth-specific evaluation metrics
    (delta1, abs_rel, rmse, silog).

    Examples:
        >>> from ultralytics.models.yolo.depth import DepthValidator
        >>> args = dict(model="yolo26n-depth.pt", data="nyu-depth.yaml")
        >>> validator = DepthValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize DepthValidator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "depth"
