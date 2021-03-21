import os
from typing import Dict, Optional, Sequence, Set, Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from trajevae.data.datasets import BaseDataset
from trajevae.utils.h36m import h36m_metadata
from trajevae.utils.skeleton import Skeleton
from typing_extensions import Literal


class DatasetH36M(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        num_steps: int,
        mode: str,
        t_his: int = 25,
        t_pred: int = 100,
        actions: Union[Set[str], Literal["all"]] = "all",
        is_valid: bool = False,
        use_vel: bool = False,
        seed: Optional[int] = None,
        is_test_run: bool = False,
        valid_step: int = 25,
        scaler: Optional[StandardScaler] = None,
        standardize_data: bool = False,
        use_augmentation: bool = True,
    ):
        self.use_vel = use_vel
        self.data_dir = data_dir

        super().__init__(
            num_steps,
            mode,
            t_his,
            t_pred,
            actions,
            is_valid=is_valid,
            seed=seed,
            is_test_run=is_test_run,
            valid_step=valid_step,
            scaler=scaler,
            standardize_data=standardize_data,
            use_augmentation=use_augmentation,
        )
        if use_vel:
            self.traj_dim += 3

        self.metadata = h36m_metadata
        self.use_augmentation = True

    def prepare_data(self):
        self.data_file = os.path.join(self.data_dir, "data_3d_h36m.npz")
        if self.is_test_run:
            self.subjects_split = {"train": [1], "test": [1]}
        else:
            self.subjects_split = {"train": [1, 5, 6, 7, 8], "test": [9, 11]}
        self.subjects = ["S%d" % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(
            parents=[
                -1,
                0,
                1,
                2,
                3,
                4,
                0,
                6,
                7,
                8,
                9,
                0,
                11,
                12,
                13,
                14,
                12,
                16,
                17,
                18,
                19,
                20,
                19,
                22,
                12,
                24,
                25,
                26,
                27,
                28,
                27,
                30,
            ],
            joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
            joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
        )
        self.removed_joints = {
            4,
            5,
            9,
            10,
            11,
            16,
            20,
            21,
            22,
            23,
            24,
            28,
            29,
            30,
            31,
        }
        self.kept_joints = np.array(
            [x for x in range(32) if x not in self.removed_joints]
        )
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)[
            "positions_3d"
        ].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != "all":
            for key in list(data_f.keys()):
                data_f[key] = dict(
                    filter(
                        lambda x: all([a in x[0] for a in self.actions]),
                        data_f[key].items(),
                    )
                )
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                data_s[action] = seq
        self.data = data_f


class DatasetH36MWithLimbTrajectories(DatasetH36M):
    def __init__(
        self,
        data_dir: str,
        num_steps: int,
        mode: str,
        joint_dropout: float,
        joint_indices_to_use: Optional[Sequence[int]] = None,
        t_his: int = 25,
        t_pred: int = 100,
        actions: Union[Set[str], Literal["all"]] = "all",
        use_vel: bool = False,
        is_valid: bool = False,
        seed: Optional[int] = None,
        is_test_run: bool = False,
        valid_step: int = 25,
        scaler: Optional[StandardScaler] = None,
        standardize_data: bool = True,
        use_augmentation: bool = True,
    ):
        super().__init__(
            data_dir,
            num_steps,
            mode,
            t_his=t_his,
            t_pred=t_pred,
            actions=actions,
            is_valid=is_valid,
            use_vel=use_vel,
            seed=seed,
            is_test_run=is_test_run,
            valid_step=valid_step,
            scaler=scaler,
            standardize_data=standardize_data,
            use_augmentation=use_augmentation,
        )
        self.joint_dropout = joint_dropout
        self.joint_indices_to_use = joint_indices_to_use

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        out = super().__getitem__(index)
        trajectory = out["shift_poses"][1:]
        if self.joint_indices_to_use is not None:
            trajectory_mask = torch.zeros(out["shift_poses"].shape[1:])
            trajectory_mask[self.joint_indices_to_use] = 1.0
        else:
            trajectory_mask = (
                torch.zeros(out["shift_poses"].shape[1:-1])
                .bernoulli_(1 - self.joint_dropout)
                .unsqueeze(dim=-1)
                .repeat((1, out["shift_poses"].shape[-1]))
            )
        trajectory = trajectory * trajectory_mask[None]
        output = {
            "trajectory": trajectory,
            "trajectory_mask": trajectory_mask,
            "index": index,
            **out,
        }
        return output


if __name__ == "__main__":
    np.random.seed(0)
    actions = {"WalkDog"}
    dataset = DatasetH36M(
        "../data/processed/human36m-3d", 100, "train", actions=actions
    )
    generator = dataset.sampling_generator()
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)
