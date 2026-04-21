#!/usr/bin/env python
"""Phase 2: Downstream evaluation with distilled backbone.

Usage:
    python run_enc_distill_phase2.py <gpu> <phase1_weights> <mode> [name] [phase1_wandb_id] [epochs] [patience]
    python run_enc_distill_phase2.py <gpu> --resume <last.pt>

    mode: "inet_finetune" (ImageNet MuSGD ft), "inet_linear_probe" (ImageNet AdamW linear probe),
          "inet_adamw_finetune" (ImageNet AdamW ft), "coco_det_finetune" (COCO detection,
          yolo26s.pt-aligned recipe), "coco_det_finetune_frozen" (COCO det, frozen backbone),
          "coco_pose_finetune" (COCO pose)

Flags:
    --resume <path>: resume from checkpoint (all modes)
    --fork_from <parent_id>:<fork_step>: wandb-fork continuation (all modes)
    --lr <val>: override lr0. For coco_det_finetune this is the RECIPE lr0 at canonical
                bs=128 and gets scaled by --batch. For other modes it is the FINAL lr0.
    --batch <int>: override batch size. For coco_det_finetune, also scales lr0, nbs, and
                warmup_epochs linearly (canonical bs=128, nbs=64 -> wd_eff and lr/sample
                stay invariant). For other modes it is applied as-is.
    --nbs <int>: explicit nbs override (coco_det_finetune; bypasses auto-scaling).
"""

import sys
from pathlib import Path

import torch

from callbacks import grad_clip, muon_w, nfs_sync, paths, wandb_config
from ultralytics import YOLO


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


_COCO_DET_MODES = ("coco_det_finetune", "coco_det_finetune_frozen")

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


def main(argv: list[str]) -> None:
    """Launch a fresh phase 2 run or resume from a checkpoint."""
    argv = argv[1:]
    argv, resume = _pop_flag(argv, "--resume")
    argv, fork_from = _pop_flag(argv, "--fork_from")
    argv, lr_override = _pop_flag(argv, "--lr")
    argv, batch_override = _pop_flag(argv, "--batch")
    argv, nbs_override = _pop_flag(argv, "--nbs")
    if resume:
        resume = paths.patch_resume(resume)
    resume_args = _load_train_args(resume) if resume else {}
    gpu = argv[0] if argv else "0"
    phase1_weights = (
        argv[1]
        if len(argv) > 1
        else resume_args.get("pretrained", "runs/classify/yolo-next-encoder/phase1-d7-dinov3-convnextb/weights/best.pt")
    )
    mode = argv[2] if len(argv) > 2 else ("inet_linear_probe" if resume_args.get("freeze") else "inet_finetune")
    name = argv[3] if len(argv) > 3 else resume_args.get("name", f"phase2-{mode}-d7")
    phase1_wandb_id = argv[4] if len(argv) > 4 else ""
    epochs = int(argv[5]) if len(argv) > 5 else None
    patience = int(argv[6]) if len(argv) > 6 else None

    if mode in ("coco_det_finetune", "coco_det_finetune_frozen", "coco_pose_finetune"):
        # Infer det/pose model from phase1 cls model (yolo26s-cls.yaml -> yolo26s.yaml / yolo26s-pose.yaml)
        cls_yaml = "yolo26s-cls.yaml"
        args_yaml = Path(phase1_weights).parent.parent / "args.yaml"
        if args_yaml.exists():
            for line in args_yaml.read_text().splitlines():
                if line.startswith("model:"):
                    cls_yaml = line.split(":", 1)[1].strip()
                    break
        model_yaml = cls_yaml.replace("-cls", "-pose") if mode == "coco_pose_finetune" else cls_yaml.replace("-cls", "")
    else:
        model_yaml = "yolo26s-cls.yaml"
    wandb_group = {"coco_det_finetune": "downstream-coco", "coco_pose_finetune": "downstream-coco-pose"}.get(mode, "downstream-imagenet")

    model = YOLO(model_yaml)
    # NOTE: C2PSA remap tested and abandoned (17.77% vs 28.02% without remap).
    # Standard pretrained= flow transfers backbone layers 0-8 via intersect_dicts.
    if mode == "inet_finetune":
        model.add_callback("on_train_start", muon_w.override(0.1))
    model.add_callback("on_train_start", grad_clip.override(1.0))
    sync_start, sync_end = nfs_sync.setup(str(paths.NFS_MIRROR_ROOT), interval_sec=paths.SYNC_INTERVAL_SEC)
    model.add_callback("on_train_start", sync_start)
    model.add_callback("on_train_end", sync_end)
    model.add_callback(
        "on_pretrain_routine_start",
        wandb_config.log_config(
            model=model_yaml,
            pretrained_from=phase1_weights,
            phase1_wandb_id=phase1_wandb_id,
            mode=mode,
            cls_to_det_remap=mode == "coco_det_finetune",
            wandb_group=wandb_group,
        ),
    )
    train_args = dict(
        pretrained=phase1_weights,
        device=gpu if mode == "coco_det_finetune" else int(gpu),
        **paths.run_paths(name),
        cos_lr=True,
        warmup_bias_lr=0,
        dropout=0,
        amp=True,
        seed=0,
        deterministic=True,
        workers=8,
    )
    if mode == "inet_linear_probe":
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
    elif mode == "inet_adamw_finetune":
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
    elif mode in _COCO_DET_MODES:
        # yolo26s.pt canonical: batch=128, nbs=64, lr0=0.00038, warmup_epochs=0.98745.
        # Scale linearly when --batch overrides canonical so lr/sample, wd_eff = wd*batch/nbs,
        # and warmup span (in samples) stay invariant.
        coco_batch = int(batch_override) if batch_override else 128
        coco_scale = coco_batch / 128.0
        coco_nbs = max(1, int(nbs_override) if nbs_override else int(round(64 * coco_scale)))
        coco_base_lr = float(lr_override) if lr_override else 0.00038
        coco_lr0 = coco_base_lr * coco_scale
        coco_warmup = 0.98745 * coco_scale
        print(
            f"[coco_det_finetune] batch={coco_batch} nbs={coco_nbs} lr0={coco_lr0:.5f} "
            f"warmup_epochs={coco_warmup:.3f} (scale={coco_scale:.2f}x vs canonical bs=128)"
        )
        train_args.update(
            data="coco.yaml",
            epochs=epochs or 70,
            batch=coco_batch,
            imgsz=640,
            nbs=coco_nbs,
            patience=patience or 100,
            lr0=coco_lr0,
            lrf=0.88219,
            momentum=0.94751,
            weight_decay=0.00027,
            warmup_epochs=coco_warmup,
            warmup_momentum=0.54064,
            warmup_bias_lr=0.05684,
            cos_lr=False,
            close_mosaic=10,
            end2end=True,
            box=9.83241,
            cls=0.64896,
            dfl=0.95824,
            pose=12.0,
            kobj=1.0,
            mosaic=0.99182,
            mixup=0.05,
            cutmix=0.00082,
            copy_paste=0.40413,
            copy_paste_mode="flip",
            scale=0.9,
            fliplr=0.30393,
            translate=0.27484,
            degrees=0.00012,
            shear=0.00136,
            hsv_h=0.01315,
            hsv_s=0.35348,
            hsv_v=0.19383,
            erasing=0.4,
            auto_augment="randaugment",
            optimizer="MuSGD",
        )
        # NOTE: sgd_w/cls_w/o2m/detach_epoch from yolo26s.pt recipe are not exposed
        # as train_args in our ultralytics checkout (cfg validator rejects). muon_w
        # is set via callback since it isn't in DEFAULT_CFG_DICT either.
        model.add_callback("on_train_start", muon_w.override(0.4355))
        if mode == "coco_det_finetune_frozen":
            train_args["freeze"] = 9
    elif mode == "coco_pose_finetune":
        train_args.update(
            data="coco-pose.yaml",
            epochs=epochs or 70,
            batch=128,
            imgsz=640,
            nbs=64,
            patience=patience or 30,
            lr0=0.00125,
            lrf=0.5,
            momentum=0.937,
            weight_decay=0.0007,
            warmup_epochs=1,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            optimizer="MuSGD",
            close_mosaic=5,
            cache="disk",
            cos_lr=False,
            pose=24,
            kobj=4.0,
            mosaic=1.0,
            mixup=0,
            copy_paste=0.0,
            scale=0.9,
            fliplr=0.5,
            degrees=0.0,
            shear=0.0,
            translate=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.4,
            auto_augment="randaugment",
        )
    else:  # inet_finetune (default)
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
    # lr/batch/nbs are handled per-mode for coco_det_finetune (scaled); for other modes
    # they apply as final values.
    if mode not in _COCO_DET_MODES:
        if lr_override:
            train_args["lr0"] = float(lr_override)
        if batch_override:
            train_args["batch"] = int(batch_override)
        if nbs_override:
            train_args["nbs"] = int(nbs_override)
    if resume:
        train_args["resume"] = resume
    if fork_from:
        parent_id, fork_step = fork_from.split(":")
        wandb_config.fork_and_attach(parent_id, int(fork_step), name)
    model.train(**train_args)


if __name__ == "__main__":
    main(sys.argv)
