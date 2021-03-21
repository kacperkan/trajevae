import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

PROJECT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"


class DatasetProcessingConfig:
    SMOOTHING = 0.5


def copy2cpu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_model_module(module_path: str) -> nn.Module:
    comps = module_path.split(".")
    package = ".".join(comps[:-1])

    module = getattr(importlib.import_module(package), comps[-1])
    return module


def get_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser("TrajeGAN")

    parser.add_argument("data_folder")
    parser.add_argument("--config_name", required=True)
    parser.add_argument("--module_name", required=True)
    parser.add_argument("--load_pretrained", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument(
        "--joint_indices_to_use",
        nargs="+",
        type=int,
        required=False,
        help="Indices of joints to use from the Human36m skeleton.",
    )
    parser.add_argument(
        "--n_visualizations",
        type=int,
        required=False,
        help="Number of first samples to visualize",
    )
    parser.add_argument(
        "--n_samples_per_visualization",
        type=int,
        required=False,
        help="Number of poses to sample for a single data point.",
    )
    parser.add_argument(
        "--without_visualization",
        action="store_true",
        help="Use this flag if only scoring should be performed.",
    )
    parser.add_argument(
        "--renderer_port",
        type=int,
        required=False,
        default=8000,
        help="Port of the rendering service",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Use expected value instead of sampling randomly from the normal "
            "distribution"
        ),
    )

    # for ablations
    parser.add_argument(
        "--disable_masking_future_poses",
        action="store_true",
        help="Whether future poses should be masked or not",
    )
    parser.add_argument(
        "--disable_dct",
        action="store_true",
        help="Disables using the DCT completely",
    )
    parser.add_argument(
        "--disable_learnable_prior",
        action="store_true",
        help="Disables learnable prior completely",
    )
    parser.add_argument(
        "--disable_data_augmentation",
        action="store_true",
        help="Disables data augmentation completely",
    )
    parser.add_argument(
        "--suffix", help="Suffix for the experiment", default="", type=str
    )
    parser.add_argument(
        "--std", help="Std value of the sampling ", default=1.0, type=float
    )
    parser.add_argument(
        "--nk", help="Number of poses to sample", default=50, type=int
    )

    return vars(parser.parse_args())


def pass_args_to_config(config: EasyDict, args: Dict[str, Any]):
    if args["disable_masking_future_poses"]:
        config.mask_future_poses = False

    if args["disable_dct"]:
        config.use_dct = False

    if args["disable_learnable_prior"]:
        config.use_learnable_prior = False
        if "kld" in config.criterions:
            config.criterions["kld"]["class"] = "KLD"

    if len(args["suffix"]) > 0:
        config.experiment_name += "-{}".format(args["suffix"])

    config.sampling_std = args["std"]
    config.deterministic = args["deterministic"]
    config.nk = args["nk"]


def load_config_and_args() -> Tuple[EasyDict, Dict[str, Any]]:
    args = get_args()
    config = getattr(
        importlib.import_module("configs.{}".format(args["config_name"])),
        "CONFIG",
    )
    pass_args_to_config(config, args)
    return config, args


def drop_config_to(config: EasyDict, path: Path):
    with open((path / "config.json").as_posix(), "w") as f:
        json.dump(dict(config), f)


def create_logger(name: str):
    # create logger
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    fh = logging.FileHandler("log.log", mode="a")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("[%(asctime)s] %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    return logger
