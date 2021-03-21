from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data.dataloader import DataLoader
import cv2
import numpy as np
import torch
import tqdm
from easydict import EasyDict
from trajevae.common import copy2cpu, load_model_module
from trajevae.data.dataloaders import get_dataloaders
from trajevae.models.trajevae import TrajeVAE
from trajevae.utils.general import load_models, powerset, tensors_to_cuda
from trajevae.utils.rendering import (
    build_humanoid,
    build_trajectory,
    colormap,
    render_with_trajectories,
    wrap_in_scene,
)
from trajevae.utils.skeleton import Skeleton
from scipy.spatial.distance import pdist, squareform

CAMERA_BASE_PARAMS = np.array((-3, -3, 3))
# FRAMES_TO_RENDER = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
FRAMES_TO_RENDER = list(range(100))

INDEX_SELECTION: Dict[int, int] = defaultdict(int)


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
    all_poses: np.ndarray, trajectories: np.ndarray, renderer_port: int
) -> np.ndarray:
    all_poses = all_poses.reshape((-1, 3))
    floor_z_coordinate = all_poses.min(axis=0)[-1]
    scene = build_trajectory(
        trajectories,
        (30 / 255, 30 / 255, 8 / 255),
        trajectory_joint_size=0.025,
    )
    camera_view = all_poses.mean(axis=0) + CAMERA_BASE_PARAMS
    camera_target = all_poses.mean(axis=0)
    camera_target[-1] = 0

    scene = wrap_in_scene(
        scene,
        camera_view,
        floor_z_coordinate,
        camera_target=camera_target,
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
    model: TrajeVAE,
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


def get_possible_trajectories(
    data_loader: DataLoader, joint_indices_to_use: Sequence[int]
) -> Tuple[
    torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    all_data = []
    all_start_poses = []
    all_trajectories = []
    all_actions = []
    all_subjects = []
    all_frame_starts = []
    all_frame_ends = []

    for data in data_loader:
        all_start_poses.append(copy2cpu(data["base_pose"]))
        all_trajectories.append(copy2cpu(data["trajectory"]))
        all_data.append(copy2cpu(data["shift_poses"]))
        all_actions.append(np.array(data["action"]))
        all_subjects.append(np.array(data["subject"]))
        all_frame_starts.append(np.array(data["frame_start"]))
        all_frame_ends.append(np.array(data["frame_end"]))

    all_data_tensor = np.concatenate(all_data, axis=0)
    all_start_poses_tensor = np.concatenate(all_start_poses, axis=0)
    all_trajectories_tensor = np.concatenate(all_trajectories, axis=0)

    all_actions_tensor = np.concatenate(all_actions, axis=0)
    all_subjects_tensor = np.concatenate(all_subjects, axis=0)
    all_frame_starts_tensor = np.concatenate(all_frame_starts, axis=0)
    all_frame_ends_tensor = np.concatenate(all_frame_ends, axis=0)

    pd = squareform(
        pdist(
            all_start_poses_tensor[:, joint_indices_to_use].reshape(
                (all_start_poses_tensor.shape[0], -1)
            )
        )
    )

    output_data = []
    output_trajectories = []
    output_actions = []
    output_subjects = []
    output_frame_starts = []
    output_frame_ends = []
    for i in range(pd.shape[0]):
        try:
            ind = np.array(np.nonzero((pd[i] < 0.1) & (pd[i] > 0)))
            index = min(INDEX_SELECTION[i], ind.shpae[1] - 1)
            ind = ind[:, [index]]
            INDEX_SELECTION[i] += 1
        except:
            ind = np.array(np.nonzero(pd[i] < 0.1))
            ind = ind[:, [INDEX_SELECTION[i]]]

        output_data.append(all_data_tensor[ind])
        output_trajectories.append(
            np.repeat(
                all_trajectories_tensor[[i]], repeats=len(ind[0]), axis=0
            )
        )

        output_actions.append(all_actions_tensor[ind])
        output_subjects.append(all_subjects_tensor[ind])
        output_frame_starts.append(all_frame_starts_tensor[ind])
        output_frame_ends.append(all_frame_ends_tensor[ind])

    output_data_np = np.concatenate(output_data, axis=0)
    output_trajectories_np = np.concatenate(output_trajectories, axis=0)
    output_actions_np = list(np.concatenate(output_actions, axis=0))

    output_subjects_np = list(np.concatenate(output_subjects, axis=0))
    output_frame_starts_np = list(np.concatenate(output_frame_starts, axis=0))
    output_frame_ends_np = list(np.concatenate(output_frame_ends, axis=0))

    return (
        torch.from_numpy(output_data_np),
        torch.from_numpy(output_trajectories_np),
        output_actions_np,
        output_subjects_np,
        output_frame_starts_np,
        output_frame_ends_np,
    )


@torch.no_grad()
def generate_sequence_for_datatype(
    model: TrajeVAE,
    dataset_split: str,
    data_path: str,
    config: EasyDict,
    device: torch.device,
    dtype: torch.dtype,
    renderer_port: int,
):
    indices = [3, 6, 16, 13]
    every_nth_sample = 200
    num_samples = 20

    output_data_folder = (
        Path("outputs")
        / "renders-several-frames-sample-vs-diff-traj"
        / f"{config.experiment_name}"
    )

    output_data_folder.mkdir(parents=True, exist_ok=True)

    set_of_indices = powerset(indices)[1:][::-1]

    for current_indices in tqdm.tqdm(set_of_indices):
        current_indices = list(current_indices)
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

        (
            shift_poses,
            trajectories_tensor,
            all_actions,
            all_subjects,
            all_frame_starts,
            all_frame_ends,
        ) = get_possible_trajectories(loader, current_indices)

        num_generated_samples = 0
        pbar = tqdm.tqdm(total=num_samples, leave=False)

        sample_counter: Dict[str, int] = defaultdict(int)
        for j, sample in enumerate(shift_poses):
            if j % every_nth_sample != 0:
                continue

            sample = tensors_to_cuda(
                {
                    "shift_poses": shift_poses[j],
                    "trajectory": trajectories_tensor[[j]],
                    "action": all_actions[j],
                    "subject": all_subjects[j],
                    "frame_start": all_frame_starts[j],
                    "frame_end": all_frame_ends[j],
                },
                device,
            )
            sample_name = get_folder_name(
                "_".join(sample["action"][0].lower().split()),
                sample["subject"][0],
                sample["frame_start"].item(),
                sample["frame_end"].item(),
                sample["trajectory"].shape[-2],
            )
            sample_name += "_{}".format(sample_counter[sample_name])

            pred = get_prediction(
                data=sample["shift_poses"],
                trajectory=sample["trajectory"],
                model=model,
                device=device,
                t_his=config.t_his,
                t_pred=config.t_pred,
                dtype=dtype,
                num=1,
                deterministic=config.deterministic,
                std=config.sampling_std,
            )
            trajectories = sample["trajectory"][:, :, current_indices]

            pred = copy2cpu(pred.transpose(0, 1))
            trajectories = copy2cpu(trajectories)
            output_full_path = output_data_folder / sample_name
            output_full_path.mkdir(exist_ok=True, parents=True)

            for sample_index, pred_pose in enumerate(pred):
                for frame_to_render in FRAMES_TO_RENDER:
                    folder = output_full_path / str(sample_index)
                    folder.mkdir(parents=True, exist_ok=True)

                    file = folder / "fake_{}_{:04d}.png".format(
                        "_".join([str(j) for j in current_indices])
                        if len(current_indices) > 0
                        else "none",
                        frame_to_render,
                    )
                    if file.exists():
                        continue
                    cur_pred_pose = pred_pose[0][frame_to_render]
                    pred_frame = process_human_single(
                        cur_pred_pose,
                        loader.dataset.skeleton,
                        trajectories[0],
                        renderer_port,
                        joints_to_emphasize=current_indices,
                    )

                    cv2.imwrite(file.as_posix(), pred_frame[..., ::-1])

            # real
            real_samples = copy2cpu(sample["shift_poses"])[0]
            for frame_to_render in tqdm.tqdm(FRAMES_TO_RENDER, leave=False):
                file = output_full_path / "real_{:04d}.png".format(
                    frame_to_render
                )
                if file.exists():
                    continue
                real_sample = real_samples[frame_to_render + 1]
                real_frame = process_human_single(
                    real_sample,
                    loader.dataset.skeleton,
                    trajectories[0],
                    renderer_port,
                )
                cv2.imwrite(file.as_posix(), real_frame[..., ::-1])

            # trajectory
            file = output_full_path / "trajectory_{}.png".format(
                "_".join([str(j) for j in current_indices])
                if len(current_indices) > 0
                else "none",
            )
            if file.exists():
                continue
            trajectories_frame = process_trajectory(
                real_samples, trajectories[0], renderer_port
            )
            cv2.imwrite(file.as_posix(), trajectories_frame[..., ::-1])

            num_generated_samples += 1
            pbar.update(1)
            if num_generated_samples > num_samples:
                break
        pbar.close()


@torch.no_grad()
def generate_sequence(
    model: TrajeVAE, data_path: str, config: EasyDict, device: torch.device
):
    generate_sequence_for_datatype(
        model, "valid", data_path, config, device, torch.float32
    )


def main():
    from trajevae.common import load_config_and_args

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
        "trajevae.models.{}".format(args["module_name"])
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
