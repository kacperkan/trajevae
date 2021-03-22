from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import tqdm
from easydict import EasyDict
from trajegan.common import copy2cpu, load_model_module
from trajegan.data.dataloaders import get_dataloaders
from trajegan.models.pose2pose import Pose2Pose
from trajegan.utils.general import load_models, tensors_to_cuda
from trajegan.utils.rendering import (
    build_humanoid,
    build_trajectory,
    colormap,
    render_with_trajectories,
    wrap_in_scene,
)
from trajegan.utils.skeleton import Skeleton

FRAMES_TO_RENDER = list(range(100))

CAMERA_BASE_PARAMS = np.array((-3, -3, 3))


def process_human_single(
    pose: np.ndarray,
    skeleton: Skeleton,
    trajectories: np.ndarray,
    renderer_port: int,
    joints_to_emphasize: Optional[Union[List[int], Tuple[int, ...]]] = None,
) -> np.ndarray:
    floor_z_coordinate = pose.min(axis=0)[-1]
    scene = build_humanoid(
        pose,
        skeleton,
        trajectories,
        human_color=colormap(8 / 255, 30 / 255, 74 / 255),
        joints_to_emphasize=joints_to_emphasize,
        color_joints_to_emphasize=(74 / 255, 30 / 255, 8 / 255),
    )
    camera_target = pose.mean(axis=0)
    camera_target[-1] = 0
    scene = wrap_in_scene(
        scene,
        (pose.mean(axis=0) + CAMERA_BASE_PARAMS),
        floor_z_coordinate,
        camera_target=camera_target,
    )
    vis = render_with_trajectories(scene, renderer_port)
    return vis


def process_trajectory(
    pose: np.ndarray, trajectories: np.ndarray, renderer_port: int
) -> np.ndarray:
    floor_z_coordinate = pose.min(axis=0)[-1]
    scene = build_trajectory(
        trajectories,
        (30 / 255, 30 / 255, 8 / 255),
        trajectory_joint_size=0.025,
    )
    scene = wrap_in_scene(
        scene,
        (pose.mean(axis=0) + CAMERA_BASE_PARAMS),
        floor_z_coordinate,
    )
    vis = render_with_trajectories(scene, renderer_port)
    return vis


def get_folder_name(
    action: str,
    subject: str,
    frame_start: int,
    frame_end: int,
    num_trajectories: int,
) -> str:
    return "{}-{}-{}-{}_{}".format(
        action, subject, frame_start, frame_end, num_trajectories
    )


def get_prediction(
    data: torch.Tensor,
    trajectory: torch.Tensor,
    model: Pose2Pose,
    device: torch.device,
    dtype: torch.dtype,
    t_his: int,
    t_pred: int,
    num: int,
    deterministic: bool = False,
    std: float = 1.0,
) -> torch.Tensor:
    poses = data.permute(1, 0, 2, 3).contiguous()

    # future_poses = poses[1:].view((poses.shape[0] - 1, poses.shape[1], -1))
    future_poses = None

    Y = model.sample(
        poses[0],
        future_poses,
        trajectory,
        poses.shape[1],
        poses.shape[0] - 1,
        device,
        dtype,
        t_pred,
        num,
        deterministic=deterministic,
        std=std,
    )
    return Y


@torch.no_grad()
def generate_sequence_for_datatype(
    model: Pose2Pose,
    dataset_split: str,
    data_path: str,
    config: EasyDict,
    device: torch.device,
    dtype: torch.dtype,
    renderer_port: int,
):
    indices = [3, 6, 16, 13, 2]

    for i in tqdm.trange(len(indices) + 1):
        current_indices = indices[:i]

        output_data_folder = (
            Path("outputs")
            / "renders-single-frame-sample-vs-traj"
            / f"{config.experiment_name}"
        )

        output_data_folder.mkdir(parents=True, exist_ok=True)

        _, loader = get_dataloaders(
            data_path=data_path,
            config=config,
            base_dataset_class_name=config.dataset_type,
            is_test_run=False,
            is_debug=False,
            actions=config.actions,
            batch_size=1,
            are_both_valid_loaders=True,
            joint_indices_to_use=current_indices,
        )

        loader_iterator = iter(loader)
        sample = next(loader_iterator)
        sample_name = get_folder_name(
            "_".join(sample["action"][0].lower().split()),
            sample["subject"][0],
            sample["frame_start"].item(),
            sample["frame_end"].item(),
            sample["trajectory"].shape[-2],
        )
        while sample_name != "eating-S9-2050-2151_17":
            sample = next(loader_iterator)
            sample_name = get_folder_name(
                "_".join(sample["action"][0].lower().split()),
                sample["subject"][0],
                sample["frame_start"].item(),
                sample["frame_end"].item(),
                sample["trajectory"].shape[-2],
            )

        sample = tensors_to_cuda(sample, device)
        pred = get_prediction(
            data=sample["shift_poses"],
            trajectory=sample["trajectory"],
            model=model,
            device=device,
            t_his=config.t_his,
            t_pred=config.t_pred,
            dtype=dtype,
            num=11 if not config.deterministic else 1,
            deterministic=config.deterministic,
            std=config.sampling_std,
        )
        trajectories = sample["trajectory"][:, :, current_indices]

        pred = copy2cpu(pred.transpose(0, 1))
        trajectories = copy2cpu(trajectories)
        output_full_path = output_data_folder / sample_name
        output_full_path.mkdir(exist_ok=True, parents=True)

        for sample_index, pred_pose in enumerate(pred):
            for frame_index in tqdm.tqdm(FRAMES_TO_RENDER, leave=False):
                pred_pose = pred[sample_index][0][frame_index]
                folder = output_full_path / str(sample_index)
                folder.mkdir(parents=True, exist_ok=True)

                file = folder / "fake_{:04d}_{}.png".format(
                    frame_index,
                    "_".join([str(j) for j in current_indices])
                    if len(current_indices) > 0
                    else "none",
                )
                if file.exists():
                    continue
                pred_frame = process_human_single(
                    pred_pose,
                    loader.dataset.skeleton,
                    trajectories[0],
                    renderer_port,
                    joints_to_emphasize=current_indices,
                )

                cv2.imwrite(file.as_posix(), pred_frame[..., ::-1])

        # real
        for frame_index in tqdm.tqdm(FRAMES_TO_RENDER, leave=False):
            real_sample = copy2cpu(sample["shift_poses"][0, frame_index + 1])
            file = output_full_path / "real_{:04d}_{}.png".format(
                frame_index,
                "_".join([str(j) for j in current_indices])
                if len(current_indices) > 0
                else "none",
            )
            if file.exists():
                continue
            real_frame = process_human_single(
                real_sample,
                loader.dataset.skeleton,
                trajectories[0],
                renderer_port,
            )
            cv2.imwrite(file.as_posix(), real_frame[..., ::-1])

        # trajectory
        trajectories_frame = process_trajectory(
            real_sample, trajectories[0], renderer_port
        )
        file = output_full_path / "trajectory_{}.png".format(
            "_".join([str(j) for j in current_indices])
            if len(current_indices) > 0
            else "none",
        )
        if file.exists():
            continue
        cv2.imwrite(file.as_posix(), trajectories_frame[..., ::-1])


@torch.no_grad()
def generate_sequence(
    model: Pose2Pose, data_path: str, config: EasyDict, device: torch.device
):
    generate_sequence_for_datatype(
        model, "valid", data_path, config, device, torch.float32
    )


def main():
    from trajegan.common import load_config_and_args

    config, args = load_config_and_args()
    config.joint_indices_to_use = (
        args["joint_indices_to_use"] or config.joint_indices_to_use
    )
    config.n_visualizations = 1
    config.n_samples_per_visualization = (
        args["n_samples_per_visualization"]
        or config.n_samples_per_visualization
    )
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = load_model_module(
        "trajegan.models.{}".format(args["module_name"])
    )(config, args["test_run"]).to(device)
    model = model.eval()

    model_path = Path("outputs") / "models" / config.experiment_name
    load_models(model_path.as_posix(), ["model"], [model])

    generate_sequence_for_datatype(
        model,
        "valid",
        args["data_folder"],
        config,
        device,
        torch.float32,
        args["renderer_port"],
    )


if __name__ == "__main__":
    main()
