from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from easydict import EasyDict
from trajevae.metrics import computer_training_metrics
from trajevae.models.layers import MLP as MLPUnit
from trajevae.models.layers import RNN as RNNUnit
from wandb.wandb_run import Run

from ._base import BaseModel


class Model(BaseModel):
    def __init__(self, config: EasyDict, is_test_run: bool) -> None:
        super().__init__(config, is_test_run)

        self.num_joints = self.config.num_joints
        self.limb_indices = self.config.limb_indices
        self.cell_type = "gru"

        self.nh_rnn = 128
        self.nh_mlp = [300, 200]
        self.e_birnn = True

        self.e_rnn = RNNUnit(
            self.num_joints * 3 * 2,
            self.nh_rnn,
            bi_dir=self.e_birnn,
            cell_type=self.cell_type,
        )
        self.e_mlp = MLPUnit(self.nh_rnn, self.nh_mlp + [self.nh_rnn])

        self.d_rnn = RNNUnit(
            self.num_joints * 3 * 2 + self.e_mlp.out_dim,
            self.nh_rnn,
            cell_type=self.cell_type,
        )
        self.d_mlp = MLPUnit(self.nh_rnn, self.nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.num_joints * 3)

        self.d_rnn.set_mode("step")

        print("Total parameters: {}".format(self.num_parameters))
        print(
            "Total trainable parameters: {}".format(
                self.num_trainable_parameters
            )
        )
        print(
            "Total non-trainable parameters: {}".format(
                self.num_non_trainable_parameters
            )
        )

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.e_birnn:
            h_x = self.e_rnn(x).mean(dim=0)
        else:
            h_x = self.e_rnn(x)[-1]
        return h_x

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_x = self.encode_x(x)
        h = self.e_mlp(h_x)
        return h, h_x

    def decode(
        self,
        x: torch.Tensor,
        trajectory: torch.Tensor,
        z: torch.Tensor,
        h_d: torch.Tensor,
    ) -> torch.Tensor:
        self.d_rnn.initialize(batch_size=x.shape[1], hx=h_d)

        y = []
        y_i: Optional[torch.Tensor] = None
        for i in range(trajectory.shape[0]):
            y_p = x if y_i is None else y_i
            rnn_in = torch.cat((z, y_p, trajectory[i]), dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h) + y_p
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(
        self,
        pose: torch.Tensor,
        future_poses: Optional[torch.Tensor],
        trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pose_rep = pose.unsqueeze(dim=0).repeat_interleave(
            len(trajectory), dim=0
        )
        inp = torch.cat((pose_rep, trajectory), dim=-1)
        z, h_d = self.encode(inp)

        poses = self.decode(pose, trajectory, z, h_d).view(
            (trajectory.shape[0], trajectory.shape[1], self.num_joints, 3)
        )
        return {"predicted_poses": poses}

    def sample(
        self,
        base_pose: torch.Tensor,
        future_poses: Optional[torch.Tensor],
        trajectories: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        time_steps: int,
        num: int,
        deterministic: bool = False,
        std: float = 1.0,
    ) -> torch.Tensor:

        base_pose = (
            base_pose.view((batch_size, -1))
            .unsqueeze(dim=1)
            .repeat((1, num, 1))
            .view((batch_size * num, -1))
        )
        trajectories = (
            trajectories.transpose(0, 1)
            .unsqueeze(dim=2)
            .repeat((1, 1, num, 1, 1))
            .view((seq_len, batch_size * num, -1))
        )

        output = self(base_pose, None, trajectories)
        output_poses = output["predicted_poses"]
        output_poses = output_poses.view(
            (time_steps, batch_size, num, self.num_joints, -1)
        ).permute((1, 2, 0, 3, 4))

        return output_poses

    def _generator_step(
        self,
        batch: Dict[str, torch.Tensor],
        config: EasyDict,
        additional_metadata: Dict[str, Any],
        criterions: Dict[str, nn.Module],
        writer: Optional[Run],
        global_step: int,
        is_valid: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        losses: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0).to(self.device)
        )

        base_pose = batch["base_pose"]
        base_pose = base_pose.view((base_pose.shape[0], -1))
        output = self(
            base_pose,
            None,
            batch["trajectory"]
            .view(
                (
                    batch["trajectory"].shape[0],
                    batch["trajectory"].shape[1],
                    -1,
                )
            )
            .transpose(0, 1),
        )

        fake_poses = output["predicted_poses"]
        real_poses = batch["shift_poses"][:, 1:].transpose(0, 1)

        losses["rec"] = criterions["rec"](real_poses, fake_poses)
        loss: torch.Tensor = sum(losses.values())

        out_metrics = {
            "generator/{}".format(key): value.item()
            for key, value in losses.items()
        }
        out_metrics["generator/total"] = loss.item()
        out_metrics.update(
            computer_training_metrics(
                # calculating without the root node
                fake_poses,
                real_poses,
                skeleton=additional_metadata["skeleton"],
            )
        )

        if writer is not None:
            prefix = "train/" if not is_valid else "valid/"
            writer.log(
                {prefix + key: val for key, val in out_metrics.items()},
                step=global_step,
            )

            if global_step % config.log_img_frequency == 0:
                self.log_images(
                    writer,
                    real_poses.transpose(0, 1),
                    fake_poses.transpose(0, 1),
                    batch["trajectory"],
                    additional_metadata,
                    global_step,
                    is_valid,
                )
        return loss, output, out_metrics
