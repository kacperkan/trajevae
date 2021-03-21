import abc
import math
from typing import Dict, Optional, Set, Union

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing_extensions import Literal


class BaseDataset(Dataset):
    def __init__(
        self,
        num_steps: int,
        mode: str,
        t_his: int,
        t_pred: int,
        actions: Union[Set[str], Literal["all"]] = "all",
        is_valid: bool = False,
        seed: Optional[int] = 1337,
        is_test_run: bool = False,
        valid_step: int = 25,
        scaler: Optional[StandardScaler] = None,
        standardize_data: bool = False,
        use_augmentation: bool = True,
    ):
        self.valid_step = valid_step
        self.is_test_run = is_test_run
        self.is_valid = is_valid
        self.num_steps = num_steps
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.data: Dict[str, Dict[str, np.ndarray]]
        self.kept_joints: Optional[np.ndarray] = None
        self.std, self.mean = None, None
        self.use_augmentation = False
        if standardize_data:
            self.scaler = scaler or StandardScaler()
        else:
            self.scaler = scaler or StandardScaler()
        self.use_augmentation = use_augmentation

        self.prepare_data()

        self.data_len = sum(
            [
                seq.shape[0]
                for data_s in self.data.values()
                for seq in data_s.values()
            ]
        )

        self.traj_dim = -1
        if self.kept_joints is not None:
            self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        # iterator specific
        self.sample_ind = None
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        if self.data is not None:
            self.subject_data = []
            for subject, data_s in self.data.items():
                for action_name, sequence in data_s.items():
                    if (
                        self.is_test_run
                        and "walking" not in action_name.lower()
                    ):
                        continue
                    seq_len = sequence.shape[0]
                    for i in range(0, seq_len - self.t_total, self.valid_step):
                        self.subject_data.append(
                            (subject, action_name, slice(i, i + self.t_total))
                        )

    @abc.abstractmethod
    def prepare_data(self):
        pass

    def set_seed(self, seed: Optional[int]):
        self._rng = np.random.RandomState(seed)
        self.seed = seed

    def sample(self):
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        fr_start = np.random.randint(seq.shape[0] - self.t_total)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start:fr_end]
        return traj[None, ...]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.data is None:
            raise ValueError("Prepare dataset first using `prepare_data()`")
        if self.is_test_run:
            index = 0
            subject, action, sequence_slice = self.subject_data[index]
            sequence = self.data[subject][action]
        elif self.is_valid:
            subject, action, sequence_slice = self.subject_data[index]
            sequence = self.data[subject][action]
        else:
            subject = np.random.choice(self.subjects)
            dict_s = self.data[subject]
            action = self._rng.choice(list(dict_s.keys()))
            sequence = dict_s[action]
            fr_start = self._rng.randint(sequence.shape[0] - self.t_total)
            fr_end = fr_start + self.t_total
            sequence_slice = slice(fr_start, fr_end)
        orig_traj = sequence[sequence_slice].copy()
        shift_traj = orig_traj.copy()
        shift_traj = shift_traj - shift_traj[:1, :1]

        if self.use_augmentation and not self.is_valid:
            angle = self._rng.uniform(0, 2 * math.pi - 1e-8)
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1.0],
                ],
            )

            shift_traj = shift_traj @ rotation_matrix.T

        shift_traj = torch.from_numpy(shift_traj).float()
        base_pose = shift_traj[0]

        return {
            "poses": orig_traj,
            "shift_poses": shift_traj,
            "action": action,
            "subject": subject,
            "base_pose": base_pose,
            "frame_start": sequence_slice.start,
            "frame_end": sequence_slice.stop,
        }

    def __len__(self) -> int:
        if self.is_valid:
            return len(self.subject_data)
        return self.num_steps

    def sampling_generator(self, num_samples: int = 1000, batch_size: int = 8):
        for _ in range(num_samples // batch_size):
            sample = []
            for _ in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            yield sample

    def iter_generator(self, step: int = 25):
        if self.data is None:
            raise ValueError("Prepare dataset first using `prepare_data()`")
        for data_s in self.data.values():
            for sequence in data_s.values():
                seq_len = sequence.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = sequence[None, i : i + self.t_total]
                    yield traj
