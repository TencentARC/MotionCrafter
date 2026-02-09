"""Evaluation entrypoint that loads gin configs and runs the trainer for a chosen dataset."""

import argparse
from accelerate.logging import get_logger
import gin
from trainers import TrainerConfig

logger = get_logger(__name__, log_level="INFO")


def main():
    """Parse args, load dataset-specific gin configs, and launch training/evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--dataset",
        choices=["scannet", "sintel", "monkaa", "vkitti2", "kubric", "spring", "dynamic_replica", "point_odyssey"],
        required=True,
        help="Choose the dataset"
    )
    args = parser.parse_args()

    # Collect inline gin bindings (highest priority)
    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)

    # Collect gin config files
    gincs = []
    if args.ginc:
        gincs.extend(args.ginc)

    # Append dataset-specific gin configs
    if args.dataset == "scannet":
        gincs.append("configs/eval_datasets/scannet.gin")
    elif args.dataset == "sintel":
        gincs.append("configs/eval_datasets/sintel.gin")
    elif args.dataset == "monkaa":
        gincs.append("configs/eval_datasets/monkaa.gin")
    elif args.dataset == "vkitti2":
        gincs.append("configs/eval_datasets/vkitti2.gin")
    elif args.dataset == "kubric":
        gincs.append("configs/eval_datasets/kubric.gin")
    elif args.dataset == "spring":
        gincs.append("configs/eval_datasets/spring.gin")
    elif args.dataset == "dynamic_replica":
        gincs.append("configs/eval_datasets/dynamic_replica.gin")
    elif args.dataset == "point_odyssey":
        gincs.append("configs/eval_datasets/point_odyssey.gin")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Parse configs and finalize
    gin.parse_config_files_and_bindings(gincs, ginbs, finalize_config=True)

    trainer_cfg = TrainerConfig()
    trainer = trainer_cfg.build()

    # Log operative config and persist it under experiment directory
    conf = gin.operative_config_str()
    logger.info(conf)
    with open(trainer_cfg.exp_path / "config_{}.gin".format(args.dataset), "w") as f:
        f.write(conf)

    trainer.train()


if __name__ == "__main__":
    main()
