from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = METHOD_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding.train_gae import train_gae_from_config
from embedding.train_gaga import train_gaga_from_config
from piuot.yaml_config import load_yaml_config, reduction_method_from_config
from project_paths import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a latent embedding with the YAML-selected reduction method (gae or gaga)."
    )
    parser.add_argument("--config", "--yaml-config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml_config(args.config_path)
    method = reduction_method_from_config(config)
    if method == "gae":
        train_gae_from_config(args.config_path)
        return
    if method == "gaga":
        train_gaga_from_config(args.config_path)
        return
    raise ValueError(f"Unsupported reduction method '{method}'.")


if __name__ == "__main__":
    main()

