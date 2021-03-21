import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from easydict import EasyDict
from trajevae.common import load_config_and_args, load_model_module
from trajevae.evaluators.main_score import compute_multimodal_trajectories
from trajevae.evaluators.main_score import (
    compute_stats as main_score_compute_stats,
)
from trajevae.utils.general import load_models


def evaluate_main_score(config: EasyDict, args: Dict[str, Any], name: str):
    original_t_his = config.t_his
    config.n_visualizations = (
        args["n_visualizations"] or config.n_visualizations
    )
    config.n_samples_per_visualization = (
        args["n_samples_per_visualization"]
        or config.n_samples_per_visualization
    )
    if config.dataset_type == "human36m":
        possible_indices_sets = [
            [],
            [3],
            [3, 6],
            [3, 6, 13],
            [3, 6, 13, 16],
        ]
    else:
        raise ValueError(
            ("Unknown dataset type: {}. Available are: human36m").format(
                config.dataset_type
            )
        )
    config.batch_size = config.validation_batch_size
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.set_grad_enabled(False)

    model = load_model_module(
        "trajegan.models.{}".format(args["module_name"])
    )(config, args["test_run"]).to(device)
    model = model.eval()

    model_path = Path("outputs") / "models" / config.experiment_name
    load_models(model_path.as_posix(), ["model"], [model])

    for index_set in possible_indices_sets:
        print("Running for indices: {}".format(index_set))
        config.joint_indices_to_use = index_set
        config.experiment_subfolder = "{}_{}".format(
            name,
            "none"
            if (
                config.joint_indices_to_use is None
                or len(config.joint_indices_to_use) == 0
            )
            else "_".join(map(str, config.joint_indices_to_use)),
        )

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        main_score_compute_stats(
            data_path=args["data_folder"],
            cfg=config,
            model=model,
            device=device,
            dtype=dtype,
            use_pseudo_past_frames=False,
            is_test_run=args["test_run"],
        )
    config.t_his = original_t_his


def evaluate_multimodal_trajectories(config: EasyDict, args: Dict[str, Any]):
    original_t_his = config.t_his
    config.n_visualizations = (
        args["n_visualizations"] or config.n_visualizations
    )
    config.n_samples_per_visualization = (
        args["n_samples_per_visualization"]
        or config.n_samples_per_visualization
    )
    if config.dataset_type == "human36m":
        possible_indices_sets = [
            [3],
            [3, 6],
            [3, 6, 13],
            [3, 6, 13, 16],
        ]
    else:
        raise ValueError(
            ("Unknown dataset type: {}. Available are: human36m").format(
                config.dataset_type
            )
        )
    config.batch_size = config.validation_batch_size
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.set_grad_enabled(False)

    model = load_model_module(
        "trajegan.models.{}".format(args["module_name"])
    )(config, args["test_run"]).to(device)
    model = model.eval()

    model_path = Path("outputs") / "models" / config.experiment_name
    load_models(model_path.as_posix(), ["model"], [model])

    for index_set in possible_indices_sets:
        print("Running for indices: {}".format(index_set))
        config.joint_indices_to_use = index_set
        config.experiment_subfolder = "{}_{}".format(
            "multimodal_traj",
            "none"
            if (
                config.joint_indices_to_use is None
                or len(config.joint_indices_to_use) == 0
            )
            else "_".join(map(str, config.joint_indices_to_use)),
        )

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        compute_multimodal_trajectories(
            data_path=args["data_folder"],
            cfg=config,
            model=model,
            device=device,
            dtype=dtype,
            is_test_run=args["test_run"],
        )
    config.t_his = original_t_his


def main():
    config, args = load_config_and_args()

    # evaluate_multimodal_trajectories(
    #     copy.deepcopy(config), copy.deepcopy(args)
    # )
    evaluate_main_score(copy.deepcopy(config), copy.deepcopy(args), "joints")


if __name__ == "__main__":
    main()
