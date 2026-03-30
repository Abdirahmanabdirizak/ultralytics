"""Callback to log custom trainer attributes to WandB config and args.yaml.

Usage:
    from callbacks import wandb_config

    # CE experiments: pass params explicitly
    model.add_callback("on_pretrain_routine_start", wandb_config.log_config(loss_mode="ce", muon_w=0.1))

    # Text-aligned experiments: picks up attrs from TextClassificationTrainer
    model.add_callback("on_pretrain_routine_start", wandb_config.log_config())
"""

from pathlib import Path

from ultralytics.utils import YAML

EXTRA_ATTRS = ("loss_mode", "muon_w", "use_clip_classifier", "teacher_variant", "teacher_temps", "grad_clip_norm")


def log_config(**extra_kv):
    """Return on_pretrain_routine_start callback to log custom config.

    Args:
        **extra_kv: Key-value pairs to set on trainer and log. For TextClassificationTrainer, these attrs already exist;
            for ClassificationTrainer, they're set via extra_kv.
    """

    def callback(trainer):
        for k, v in extra_kv.items():
            if not hasattr(trainer, k):
                setattr(trainer, k, v)
        config = {k: getattr(trainer, k) for k in EXTRA_ATTRS if hasattr(trainer, k)}
        config.update(extra_kv)
        # Update args.yaml
        args_path = Path(trainer.save_dir) / "args.yaml"
        if args_path.exists() and config:
            data = YAML.load(args_path)
            data.update(config)
            YAML.save(args_path, data)
        # Update WandB
        try:
            import wandb

            if wandb.run and config:
                wandb.run.config.update(config, allow_val_change=True)
        except ImportError:
            pass

    return callback
