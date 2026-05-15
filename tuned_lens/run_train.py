"""
Entry point for tuned lens training.

Usage examples:
    # Use the default YAML config
    python run_train.py

    # Use a different YAML config
    python run_train.py --config configs/tuned_lens/my_config.yaml

    # Override individual fields on the command line (takes precedence over YAML)
    python run_train.py --layers 1 5 10 20 27 --batch_size 8 --num_epochs 5

    # Combine a YAML config with CLI overrides
    python run_train.py --config configs/tuned_lens/default.yaml --learning_rate 5e-4
"""

import argparse
import sys
import os

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from config import TunedLensConfig
from train import train

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "tuned_lens", "default.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train tuned lens for VLMs")

    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to a YAML config file (default: configs/tuned_lens/default.yaml)."
    )

    # All TunedLensConfig fields are also available as CLI args; they override the YAML
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_config", type=str)
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lambda_reg", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--tensorboard_dir", type=str)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    # Start with dataclass defaults
    config = TunedLensConfig()

    # Apply YAML config
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    for key, value in yaml_cfg.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: unknown config key '{key}' in {args.config}")

    # CLI overrides (only fields explicitly set on the command line)
    cli_fields = [
        "model_name", "dataset_name", "dataset_config", "layers",
        "seq_len", "batch_size", "num_epochs", "learning_rate",
        "lambda_reg", "output_dir", "tensorboard_dir", "log_every", "dtype", "device",
    ]
    for field in cli_fields:
        value = getattr(args, field, None)
        if value is not None:
            setattr(config, field, value)

    print("Config:")
    for field in sorted(cli_fields + ["eval_every_epoch"]):
        print(f"  {field}: {getattr(config, field)}")
    print()

    train(config)


if __name__ == "__main__":
    main()
