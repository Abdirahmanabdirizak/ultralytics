from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import copy_attr
from .tasks import load_checkpoint
import torch.nn.functional as F
from torch import nn
import torch


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict, idx):
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module, input, output):
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted
    from both models via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Layer indices for feature extraction.
        projector (nn.ModuleList): MLP projector aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.
        split_sizes (list): Spatial sizes for splitting teacher score maps per feature level.

    Methods:
        __init__: Initialize the distillation model with teacher, student, and feature layers.
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        train: Set training mode while keeping teacher frozen.
        fuse: Fuse model layers for inference speedup.
        load_from_module: Load distillation weights from a checkpoint.

    Examples:
        Initialize a distillation model
        >>> from ultralytics.nn.distill_model import DistillationModel
        >>> model = DistillationModel(teacher_model="yolo26s.pt", student_model="yolo26n.pt")
    """

    def __init__(self, teacher_model: str | nn.Module, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        if isinstance(teacher_model, str):
            teacher_model = load_checkpoint(teacher_model)[0]
        feats_idx = self.get_distill_layers(student_model)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = feats_idx

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats = {}
        self._student_feats = {}
        self._hooks = []
        for idx in feats_idx:
            self._hooks.append(
                self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
            )
            self._hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.imgsz
        with torch.inference_mode():
            teacher_model(torch.zeros(2, 3, imgsz, imgsz).to(device))
            student_model(torch.zeros(2, 3, imgsz, imgsz).to(device))
        teacher_output = [self._teacher_feats[idx] for idx in feats_idx]
        student_output = [self._student_feats[idx] for idx in feats_idx]
        assert len(teacher_output) == len(student_output), "Feature dimensions must match in length."

        self.split_sizes = []
        for tf in teacher_output[:-1]:
            f = self.decouple_outputs(tf)
            if not isinstance(f, dict):
                self.split_sizes.append(f.shape[-2] * f.shape[-1])
        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out, shape_check=True).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out, shape_check=True).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors)

    def __getstate__(self):
        """Return a clean copy of state for pickling without hooks and extracted features."""
        state = self.__dict__.copy()
        state["_hooks"] = []
        state["_teacher_feats"] = {}
        state["_student_feats"] = {}
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._hooks = []
        for idx in self.feats_idx:
            # Clear stale hooks that were pickled with the submodels
            self.teacher_model.model[idx]._forward_hooks.clear()
            self.student_model.model[idx]._forward_hooks.clear()
            self._hooks.append(
                self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
            )
            self._hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )

    @staticmethod
    def get_distill_layers(model):
        """Auto-detect distillation feature layers from the model's Detect head.

        Returns the Detect head's input layer indices plus the head layer index itself.
        E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].
        """
        for m in model.model:
            if isinstance(m, Detect):  # covers Detect, Segment, Pose, OBB, etc.
                return list(m.f) + [m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed for distillation."""
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not self.training:
            preds = self.student_model(batch["img"])
            regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
            loss_distill = torch.zeros(1, device=batch["img"].device)
            distill_loss_detach = torch.zeros(1, device=batch["img"].device)
            return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.inference_mode():
            self.teacher_model(batch["img"])  # hooks capture teacher features
        preds = self.student_model(batch["img"])  # hooks capture student features

        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)

        distill_loss = torch.zeros(1, device=batch["img"].device)
        teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        parts = torch.split(teacher_scores, self.split_sizes, dim=-1)
        teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            teacher_feat = self.decouple_outputs(self._teacher_feats[feat_idx])
            student_feat = self.decouple_outputs(self._student_feats[feat_idx])
            if isinstance(teacher_feat, dict):
                continue
            student_feat = (
                self.projector[i](student_feat)
                if student_feat.ndim == 4
                else student_feat
            )
            distill_loss += self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis

        distill_loss_detach = distill_loss.detach()
        batch_size = batch["img"].shape[0]
        loss_distill = distill_loss * batch_size
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])

    def loss_sl2(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        feat_idx: int = 0,
        teacher_scores: tuple | None = None,
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple | None): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        if hasattr(self.student_model, "set_head_attr"):
            self.student_model.set_head_attr(**kwargs)

    def fuse(self, verbose: bool = True):
        """Fuse model layers for inference speedup."""
        self.student_model.fuse(verbose)
        return self

    def load_from_module(self, weights: dict | nn.Module, strict: bool = False):
        """Load distillation weights from a checkpoint dict or module."""
        module = weights["model"] if isinstance(weights, dict) else weights
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module or checkpoint dict, got {type(weights).__name__}")
        incompatible = self.load_state_dict(module.float().state_dict(), strict=strict)
        self._freeze_teacher()
        return incompatible

    def decouple_outputs(self, preds, shape_check=False, branch="one2one"):
        """Decouple outputs for teacher/student models."""
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict):
            if branch in preds:
                preds = preds[branch]
            if shape_check:
                preds = preds["boxes"]
        return preds
