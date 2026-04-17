#!/usr/bin/env python
"""Phase 2: Downstream evaluation with distilled backbone.

Usage:
    python run_enc_distill_phase2.py <gpu> <phase1_weights> <mode> [name] [phase1_wandb_id] [epochs] [patience]
    python run_enc_distill_phase2.py <gpu> --resume <last.pt>

    mode: "finetune" (MuSGD), "linear" (AdamW frozen), "adamw_ft" (AdamW finetune), "coco_det" (COCO detection)

Finetune params match exp5b (reproduced CE baseline, 75.95% top-1) exactly,
only epochs/patience shortened for faster evaluation.
"""

import sys
from pathlib import Path

import torch

from callbacks import grad_clip, muon_w, nfs_sync, wandb_config
from ultralytics import YOLO


def _pop_resume(argv: list[str]) -> tuple[list[str], str]:
    """Return argv without '--resume <path>' and the resume path."""
    if "--resume" not in argv:
        return argv, ""
    index = argv.index("--resume")
    return argv[:index] + argv[index + 2 :], argv[index + 1]


def _load_train_args(resume: str) -> dict:
    """Load saved training arguments from a checkpoint."""
    return torch.load(Path(resume), map_location="cpu", weights_only=False)["train_args"]


_AUG_ARGS = dict(
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1,
    auto_augment="randaugment",
    erasing=0.4,
    crop_fraction=1,
)

# Write checkpoints/logs to local SSD, mirror to shared NFS periodically (see callbacks/nfs_sync.py).
LOCAL_PROJECT = "/home/fatih/runs/yolo-next-encoder"
NFS_MIRROR_ROOT = "/data/shared-datasets/fatih-runs/classify/yolo-next-encoder"
SYNC_INTERVAL_SEC = 600


def main(argv: list[str]) -> None:
    """Launch a fresh phase 2 run or resume from a checkpoint."""
    argv, resume = _pop_resume(argv[1:])
    resume_args = _load_train_args(resume) if resume else {}
    gpu = argv[0] if argv else "0"
    phase1_weights = (
        argv[1]
        if len(argv) > 1
        else resume_args.get("pretrained", "runs/classify/yolo-next-encoder/phase1-d7-dinov3-convnextb/weights/best.pt")
    )
    mode = argv[2] if len(argv) > 2 else ("linear" if resume_args.get("freeze") else "finetune")
    name = argv[3] if len(argv) > 3 else resume_args.get("name", f"phase2-{mode}-d7")
    phase1_wandb_id = argv[4] if len(argv) > 4 else ""
    epochs = int(argv[5]) if len(argv) > 5 else None
    patience = int(argv[6]) if len(argv) > 6 else None

    if mode in ("coco_det", "coco_det_frozen"):
        # Infer det model from phase1 cls model (e.g. yolo26s-cls.yaml -> yolo26s.yaml)
        cls_yaml = "yolo26s-cls.yaml"
        args_yaml = Path(phase1_weights).parent.parent / "args.yaml"
        if args_yaml.exists():
            for line in args_yaml.read_text().splitlines():
                if line.startswith("model:"):
                    cls_yaml = line.split(":", 1)[1].strip()
                    break
        model_yaml = cls_yaml.replace("-cls", "")
    else:
        model_yaml = "yolo26s-cls.yaml"
    wandb_group = "downstream-coco" if mode == "coco_det" else "downstream-imagenet"

    model = YOLO(model_yaml)
    # NOTE: C2PSA remap tested and abandoned (17.77% vs 28.02% without remap).
    # Standard pretrained= flow transfers backbone layers 0-8 via intersect_dicts.
    if mode in ("finetune", "coco_det", "coco_det_frozen"):
        model.add_callback("on_train_start", muon_w.override(0.1))
    model.add_callback("on_train_start", grad_clip.override(1.0))
    sync_start, sync_end = nfs_sync.setup(NFS_MIRROR_ROOT, interval_sec=SYNC_INTERVAL_SEC)
    model.add_callback("on_train_start", sync_start)
    model.add_callback("on_train_end", sync_end)
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            pretrained_from=phase1_weights,
            phase1_wandb_id=phase1_wandb_id,
            mode=mode,
            cls_to_det_remap=mode == "coco_det",
            wandb_group=wandb_group,
        ),
    )
    train_args = dict(
        pretrained=phase1_weights,
        device=gpu if mode == "coco_det" else int(gpu),
        project=LOCAL_PROJECT,
        name=name,
        cos_lr=True,
        warmup_bias_lr=0,
        dropout=0,
        amp=True,
        seed=0,
        deterministic=True,
        workers=8,
    )
    if mode == "linear":
        train_args.update(
            data="/data/shared-datasets/imagenet",
            epochs=epochs or 50,
            batch=256,
            imgsz=224,
            nbs=256,
            freeze=10,
            patience=patience or 10,
            lr0=1e-3,
            lrf=0.01,
            weight_decay=1e-3,
            warmup_epochs=1,
            optimizer="AdamW",
        )
    elif mode == "adamw_ft":
        train_args.update(
            data="/data/shared-datasets/imagenet",
            epochs=epochs or 50,
            batch=256,
            imgsz=224,
            nbs=256,
            patience=patience or 30,
            lr0=1e-3,
            lrf=0.01,
            weight_decay=1e-3,
            warmup_epochs=5,
            momentum=0.9,
            optimizer="AdamW",
            **_AUG_ARGS,
        )
    elif mode in ("coco_det", "coco_det_frozen"):
        train_args.update(
            data="coco.yaml",
            epochs=epochs or 70,
            batch=128,
            imgsz=640,
            patience=patience or 50,
            lr0=0.00038,
            lrf=0.882,
            momentum=0.948,
            weight_decay=0.00027,
            warmup_epochs=0.99,
            close_mosaic=10,
            end2end=True,
            mosaic=0.992,
            mixup=0.05,
            copy_paste=0.404,
            scale=0.9,
            fliplr=0.304,
            optimizer="MuSGD",
        )
        if mode == "coco_det_frozen":
            train_args["freeze"] = 9  # freeze backbone layers 0-8
    else:  # finetune (default)
        train_args.update(
            data="/data/shared-datasets/imagenet",
            epochs=epochs or 50,
            batch=256,
            imgsz=224,
            nbs=256,
            patience=patience or 30,
            lr0=0.1,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            warmup_epochs=0,
            optimizer="MuSGD",
            **_AUG_ARGS,
        )
    if resume:
        train_args["resume"] = resume
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
