#!/usr/bin/env python
"""Phase 1: Encoder distillation pretraining on DataComp-12M."""

import sys
from pathlib import Path

import torch

from callbacks import beta2_override, grad_clip, nfs_sync, wandb_config
from ultralytics import YOLO
from ultralytics.models.yolo.classify.train_image_encoder import ImageEncoderTrainer

RECIPES = {
    "default": dict(lr0=3e-4, weight_decay=0.05, warmup_epochs=1, epochs=10, momentum=0.9, grad_clip=3.0, beta2=None),
    # EUPE Stage 2: proxy->student distillation (arXiv:2603.22387 Sec 4.1, ssl_default_config.yaml:131-147)
    # Same loss as ours (0.9cos+0.1L1, Eq.5-6). beta2=None -> uses default 0.999 matching EUPE
    "eupe": dict(lr0=2e-5, weight_decay=1e-4, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=3.0, beta2=None),
    # AM-RADIO: multi-teacher distillation (arXiv:2312.06709 Sec 4, Eq.2-3)
    # Same loss as ours (0.9cos+0.1L1). beta2=0.95 from MobileCLIP2 (training/configs/run_dfndr2b.sh)
    "radio": dict(lr0=1e-3, weight_decay=0.02, warmup_epochs=1, epochs=30, momentum=0.9, grad_clip=1.0, beta2=0.95),
}

# Write checkpoints/logs to local SSD, mirror to shared NFS periodically (see callbacks/nfs_sync.py).
LOCAL_PROJECT = "/home/fatih/runs/yolo-next-encoder"
NFS_MIRROR_ROOT = "/data/shared-datasets/fatih-runs/classify/yolo-next-encoder"
SYNC_INTERVAL_SEC = 600


def _pop_flag(argv: list[str], flag: str, is_bool: bool = False) -> tuple[list[str], str]:
    """Pop a --flag [value] pair from argv, return (remaining_argv, value).

    Args:
        argv: argument list
        flag: flag name (e.g. "--resume")
        is_bool: if True, flag has no value argument
    """
    if flag not in argv:
        return argv, ""
    i = argv.index(flag)
    if is_bool:
        return argv[:i] + argv[i + 1 :], "true"
    return argv[:i] + argv[i + 2 :], argv[i + 1]


def _load_train_args(resume: str) -> dict:
    """Load saved training arguments from a checkpoint."""
    return torch.load(Path(resume), map_location="cpu", weights_only=False)["train_args"]


def main(argv: list[str]) -> None:
    """Launch a fresh phase 1 run or resume from a checkpoint.

    Args:
        argv: [gpu, teachers, name, recipe, model_yaml, data, epochs]
        --resume <path>: resume from checkpoint
        --cos_weight <float>: cosine loss weight (default 0.9)
        --l1_weight <float>: smooth L1 loss weight (default 0.1)
        --cls_l1: add smooth L1 to CLS token loss (default False)
    """
    args = argv[1:]
    args, resume = _pop_flag(args, "--resume")
    args, cos_w = _pop_flag(args, "--cos_weight")
    args, l1_w = _pop_flag(args, "--l1_weight")
    args, cls_l1_str = _pop_flag(args, "--cls_l1", is_bool=True)
    args, lr_override = _pop_flag(args, "--lr")

    cos_weight = float(cos_w) if cos_w else 0.9
    l1_weight = float(l1_w) if l1_w else 0.1
    cls_l1 = bool(cls_l1_str)
    lr0 = float(lr_override) if lr_override else r["lr0"]

    resume_args = _load_train_args(resume) if resume else {}
    gpu = args[0] if args else "0"
    teachers = args[1] if len(args) > 1 else resume_args.get("teachers", "eupe:vitb16")
    name = (
        args[2] if len(args) > 2 else resume_args.get("name", f"phase1-{teachers.replace(':', '-').replace('+', '_')}")
    )
    recipe = args[3] if len(args) > 3 else "default"
    model_yaml = args[4] if len(args) > 4 else "yolo26s-cls.yaml"
    data = args[5] if len(args) > 5 else "/data/shared-datasets/datacomp-12m"
    epochs = int(args[6]) if len(args) > 6 else None
    r = RECIPES[recipe]

    model = YOLO(model_yaml)
    if r["grad_clip"]:
        model.add_callback("on_train_start", grad_clip.override(r["grad_clip"]))
    if r["beta2"]:
        model.add_callback("on_train_start", beta2_override.override(r["beta2"]))
    sync_start, sync_end = nfs_sync.setup(NFS_MIRROR_ROOT, interval_sec=SYNC_INTERVAL_SEC)
    model.add_callback("on_train_start", sync_start)
    model.add_callback("on_train_end", sync_end)
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            teachers=teachers,
            recipe=recipe,
            cos_weight=cos_weight,
            l1_weight=l1_weight,
            cls_l1=cls_l1,
            grad_clip=r["grad_clip"],
            beta2=r["beta2"],
            wandb_group="distill",
        ),
    )
    train_args = dict(
        trainer=ImageEncoderTrainer,
        teachers=teachers,
        data=data,
        knn_eval="/data/shared-datasets/imagenet",
        cos_weight=cos_weight,
        l1_weight=l1_weight,
        cls_l1=cls_l1,
        device=gpu,
        project=LOCAL_PROJECT,
        name=name,
        epochs=epochs or r["epochs"],
        batch=128,
        imgsz=224,
        patience=5,
        nbs=512,
        cos_lr=True,
        lr0=lr0,
        lrf=0.01,
        momentum=r["momentum"],
        weight_decay=r["weight_decay"],
        warmup_epochs=r["warmup_epochs"],
        warmup_bias_lr=0,
        dropout=0,
        optimizer="AdamW",
        pretrained=False,
        amp=True,
        seed=0,
        deterministic=True,
        fliplr=0.5,
        workers=8,
    )
    if resume:
        train_args["resume"] = resume
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
