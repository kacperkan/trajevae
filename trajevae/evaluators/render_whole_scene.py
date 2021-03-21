import math
import random
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import seaborn as sns
import torch
import tqdm
from easydict import EasyDict
from torch.utils.data.dataloader import DataLoader
from trajevae.common import copy2cpu, load_model_module
from trajevae.data.dataloaders import get_dataloaders
from trajevae.models.trajevae import TrajeVAE
from trajevae.utils.general import load_models, tensors_to_cuda
from trajevae.utils.rendering import (
    build_humanoid,
    render_with_trajectories,
    wrap_in_scene,
)

CAMERA_BASE_PARAMS = np.array((-17, 10, 13))
FRAMES_TO_RENDER = list(range(100))
# FRAMES_TO_RENDER = [40]


class Person:
    def __init__(
        self,
        angle: float,
        shift_x: float,
        shift_y: float,
        action: str,
        indices_to_use: List[int],
        deterministic: bool = False,
    ) -> None:
        self.angle = angle
        self.action = action
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.indices_to_use = indices_to_use
        self.deterministic = deterministic

    def get_from_data(self, actions: List[str]) -> List[int]:
        indices = []
        for index, act in enumerate(actions):
            if act == self.action:
                indices.append(index)
        return indices

    def transform_pose(
        self, poses: torch.Tensor, shift_z: float
    ) -> torch.Tensor:
        rotation_matrix = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle), 0.0],
                [np.sin(self.angle), np.cos(self.angle), 0],
                [0, 0, 1.0],
            ]
        )
        translation = np.array([self.shift_x, self.shift_y, shift_z])
        return poses @ rotation_matrix.T + translation

    def get_joint_indices(self) -> List[int]:
        if self.deterministic:
            return self.indices_to_use

        size = np.random.randint(0, len(self.indices_to_use))
        return np.random.choice(self.indices_to_use, size=size, replace=False)


SMOKERS_SHIFT_X = 1.5
SMOKERS_SHIFT_Y = -0.2


PEOPLE = [
    Person(
        0.0, 1.5 + SMOKERS_SHIFT_X, -0.5 - SMOKERS_SHIFT_Y, "Smoking", [13, 16]
    ),
    Person(
        -math.pi,
        0.5 + SMOKERS_SHIFT_X,
        -1.5 - SMOKERS_SHIFT_Y,
        "Smoking",
        [13, 16],
    ),
    Person(
        0.0,
        0.5 + SMOKERS_SHIFT_X,
        0.35 - SMOKERS_SHIFT_Y,
        "Smoking",
        [13, 16],
    ),
    Person(
        2.0,
        -0.4 + SMOKERS_SHIFT_X,
        -0.55 - SMOKERS_SHIFT_Y,
        "Smoking",
        [13, 16],
    ),
    Person(0.0, -1.15, -0.15, "WalkDog 1", [3, 6, 13, 16, 0]),
    Person(0.0, 0.33, 2.0, "Posing", [3, 6, 13, 16]),
    Person(-math.pi / 4 * 3, 0.33, 1.0, "Photo 1", [3, 6, 13, 16]),
    Person(-math.pi / 2, 0.85, -1.35, "Walking 1", [0, 3, 6, 13, 16]),
    Person(0.0, -1, -1.35, "Walking", [0, 3, 6, 13, 16]),
    Person(0.0, 0.0, 0.0, "Phoning 1", [0, 3, 6, 13, 16]),
    Person(0.0, 2.4, 1.4, "Greeting", [13, 16], deterministic=True),
    Person(0.0, 0.5, -2.3, "WalkTogether", [0], deterministic=True),
    Person(0.0, 2.0, 3.0, "WalkDog", [3, 13], deterministic=True),
    Person(0.0, 3.5, 2.5, "Phoning 3", [13], deterministic=True),
    Person(0.0, 1.0, 3.2, "SittingDown 1", [3, 6], deterministic=True),
    Person(0.0, -3.0, -3.0, "Waiting 1", [16], deterministic=True),
    Person(0.0, -1.4, -4.3, "Discussion 1", [13], deterministic=True),
    Person(0.0, -0.2, -4.2, "Eating", [0, 16], deterministic=True),
]
PEOPLE_COLORS = sns.color_palette("hls", len(PEOPLE))


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


def get_sample_from_loader_with_index(
    loader: DataLoader, index: int
) -> Dict[str, torch.Tensor]:
    for cur_index, sample in enumerate(loader):
        if cur_index == index:
            return sample
    raise ValueError("Invalid index: {} for {}".format(index, len(loader)))


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
    output_data_folder = (
        Path("outputs")
        / "scene-renders"
        / f"{config.experiment_name}"
        / dataset_split
    )

    if output_data_folder.exists():
        shutil.rmtree(output_data_folder.as_posix())
    output_data_folder.mkdir(parents=True)

    _, loader = get_dataloaders(
        data_path=data_path,
        config=config,
        base_dataset_class_name=config.dataset_type,
        is_test_run=False,
        is_debug=False,
        actions=config.actions,
        batch_size=1,
        are_both_valid_loaders=True,
        joint_indices_to_use=config.joint_indices_to_use,
    )

    poses = []
    associated_trajectories = []

    data_samples = []
    for sample in tqdm.tqdm(loader):
        data_samples.append(sample)
    actions = [sample["action"][0] for sample in data_samples]
    print(set(actions))
    floor_z_coordinate = np.inf
    for person in PEOPLE:
        indices = person.get_from_data(actions)
        single_index = np.random.choice(indices)
        sample = data_samples[single_index]
        joint_indices = person.get_joint_indices()
        _, local_loader = get_dataloaders(
            data_path=data_path,
            config=config,
            base_dataset_class_name=config.dataset_type,
            is_test_run=False,
            is_debug=False,
            actions=config.actions,
            batch_size=1,
            are_both_valid_loaders=True,
            joint_indices_to_use=joint_indices,
        )
        proper_sample = get_sample_from_loader_with_index(
            local_loader, single_index
        )
        proper_sample = tensors_to_cuda(proper_sample, device)

        pred = get_prediction(
            data=proper_sample["shift_poses"],
            trajectory=proper_sample["trajectory"],
            model=model,
            device=device,
            t_his=config.t_his,
            t_pred=config.t_pred,
            dtype=dtype,
            num=1,
            deterministic=config.deterministic,
            std=config.sampling_std,
        )[0, 0]
        trajectories = proper_sample["trajectory"][0, :, joint_indices]

        pred = copy2cpu(pred)
        trajectories = copy2cpu(trajectories)

        floor_z_coordinate = min(pred[..., -1].min(), floor_z_coordinate)

        poses.append(pred)
        associated_trajectories.append(trajectories)

    for i in tqdm.tqdm(FRAMES_TO_RENDER):
        scene_components = []
        for person_i, (p, t, person) in enumerate(
            zip(poses, associated_trajectories, PEOPLE)
        ):
            shift_z = floor_z_coordinate - p[..., -1].min()
            p_transformed = person.transform_pose(p, shift_z)[i]
            t_transformed = person.transform_pose(t, shift_z)
            person_color = PEOPLE_COLORS[person_i]

            scene_components.extend(
                build_humanoid(
                    p_transformed,
                    loader.dataset.skeleton,
                    t_transformed,
                    human_color=person_color,
                    trajectory_color=(30 / 255, 30 / 255, 8 / 255),
                )
            )

        scene = wrap_in_scene(
            scene_components,
            CAMERA_BASE_PARAMS,
            floor_z_coordinate,
            resolution=(2880, 900),
        )
        vis = render_with_trajectories(scene, renderer_port)
        file = output_data_folder / "{:04d}.png".format(i)
        cv2.imwrite(file.as_posix(), vis[..., ::-1])


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
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    config.joint_indices_to_use = (
        args["joint_indices_to_use"] or config.joint_indices_to_use
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
