"""Fine-tune JAX pi0.5 on a local ERSIBench G2 LeRobot v2.1 dataset."""

import argparse
import dataclasses
import logging
import math
import pathlib
from typing import cast

from openpi.ersi import dataset
from openpi.ersi import g2
from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders
from scripts import train


def _params_dir(checkpoint: pathlib.Path) -> pathlib.Path:
    path = checkpoint.resolve()
    if (path / "params" / "_METADATA").is_file():
        return path / "params"
    if (path / "_METADATA").is_file():
        return path
    raise FileNotFoundError(f"No Orbax params found at {path}. Pass a checkpoint or its params directory.")


def main() -> None:
    base = cast(config.ERSIG2TrainConfig, config.get_config(g2.CONFIG_NAME))
    schedule = cast(optimizer.CosineDecaySchedule, base.lr_schedule)
    adamw = cast(optimizer.AdamW, base.optimizer)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--assets-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--init-from", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=base.batch_size)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save after each interval of updates. Use 0 for the final checkpoint only.",
    )
    parser.add_argument("--keep-period", type=int, help="Defaults to the effective save interval.")
    parser.add_argument("--peak-lr", type=float, default=schedule.peak_lr)
    parser.add_argument("--warmup-steps", type=int, default=schedule.warmup_steps)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--decay-lr", type=float, help="Defaults to 10%% of the peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=adamw.weight_decay)
    parser.add_argument("--clip-grad", type=float, default=adamw.clip_gradient_norm)
    parser.add_argument("--adam-beta1", type=float, default=adamw.b1)
    parser.add_argument("--adam-beta2", type=float, default=adamw.b2)
    parser.add_argument("--num-workers", type=int, default=base.num_workers)
    parser.add_argument("--fsdp-devices", type=int, default=base.fsdp_devices)
    parser.add_argument("--seed", type=int, default=base.seed)
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.fsdp_devices) < 1:
        parser.error("steps, batch size and FSDP device count must be positive.")
    if min(args.warmup_steps, args.save_interval, args.num_workers, args.seed) < 0:
        parser.error("warmup steps, save interval, data worker count and seed must be nonnegative.")
    if args.keep_period is not None and args.keep_period < 1:
        parser.error("keep period must be positive.")
    if args.decay_steps is not None and args.decay_steps <= args.warmup_steps:
        parser.error("decay steps must exceed warmup steps.")
    if not (math.isfinite(args.peak_lr) and args.peak_lr > 0):
        parser.error("peak learning rate must be finite and positive.")
    decay_lr = args.peak_lr * 0.1 if args.decay_lr is None else args.decay_lr
    if not (math.isfinite(decay_lr) and decay_lr >= 0):
        parser.error("decay learning rate must be finite and nonnegative.")
    if not (math.isfinite(args.weight_decay) and args.weight_decay >= 0):
        parser.error("weight decay must be finite and nonnegative.")
    if not (math.isfinite(args.clip_grad) and args.clip_grad > 0):
        parser.error("gradient clipping norm must be finite and positive.")
    if not (0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
        parser.error("Adam beta values must be in [0, 1).")

    root = args.dataset.resolve()
    assets_dir = args.assets_dir.resolve()
    stats = assets_dir / g2.ASSET_ID / "norm_stats.json"
    if not stats.is_file():
        parser.error(f"Normalization stats are missing at {stats}. Run scripts.compute_ersi_norm_stats first.")
    if dataset.load_metadata(root).total_frames < args.batch_size:
        parser.error("Batch size exceeds the number of frames in the dataset.")

    save_interval = args.save_interval or args.steps
    output_dir = args.output_dir.resolve()
    train_config = dataclasses.replace(
        base,
        exp_name=output_dir.name,
        output_dir=str(output_dir),
        data=dataclasses.replace(
            base.data,
            lerobot_root=str(root),
            assets=config.AssetsConfig(assets_dir=str(assets_dir), asset_id=g2.ASSET_ID),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_params_dir(args.init_from))),
        lr_schedule=dataclasses.replace(
            schedule,
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.decay_steps or max(args.steps, args.warmup_steps + 1),
            decay_lr=decay_lr,
        ),
        optimizer=dataclasses.replace(
            adamw,
            weight_decay=args.weight_decay,
            clip_gradient_norm=args.clip_grad,
            b1=args.adam_beta1,
            b2=args.adam_beta2,
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        save_interval=save_interval,
        keep_period=args.keep_period or save_interval,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        seed=args.seed,
    )
    logging.basicConfig()
    train.main(train_config)


if __name__ == "__main__":
    main()
