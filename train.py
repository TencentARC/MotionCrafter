"""Training entrypoint that loads gin configs and launches the trainer."""

import argparse
from accelerate.logging import get_logger
import gin
from trainers import TrainerConfig

logger = get_logger(__name__, log_level="INFO")


def main():
    """Parse gin configs/bindings, build trainer, and start training."""
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
    args = parser.parse_args()

    # Collect inline gin bindings (highest priority)
    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    # Parse gin config files and bindings
    gin.parse_config_files_and_bindings(args.ginc, ginbs, finalize_config=True)

    trainer_cfg = TrainerConfig()
    trainer = trainer_cfg.build()

    # Log operative config and persist to experiment folder
    conf = gin.operative_config_str()
    logger.info(conf)
    with open(trainer_cfg.exp_path / "config.gin", "w") as f:
        f.write(conf)

    trainer.train()


if __name__ == "__main__":
    main()
