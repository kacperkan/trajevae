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
        self.x_birnn = True
        self.nz = 256
        self.use_drnn_mlp = True

        self.x_rnn = RNNUnit(
            self.num_joints * 3,
            self.nh_rnn,
            bi_dir=self.e_birnn,
            cell_type=self.cell_type,
        )
        self.e_rnn = RNNUnit(
            self.num_joints * 3,
            self.nh_rnn,
            bi_dir=self.e_birnn,
            cell_type=self.cell_type,
        )

        self.x_mlp = MLPUnit(self.nh_rnn, self.nh_mlp + [self.nh_rnn])
        self.e_mlp = MLPUnit(self.nh_rnn * 2, self.nh_mlp + [self.nh_rnn])

        self.e_mu = nn.Linear(self.e_mlp.out_dim, self.nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, self.nz)

        self.drnn_mlp = MLPUnit(
            self.nh_rnn, self.nh_mlp + [self.nh_rnn], activation="tanh"
        )

        self.d_rnn = RNNUnit(
            self.num_joints * 3 * 2 + self.nz + self.nh_rnn,
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
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)[-1]
        return h_y

    def encode(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def reparametrize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn_like(mu)
        std = logvar.mul(0.5).exp()
        return noise * std + mu

    def decode(
        self, base_pose: torch.Tensor, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        time_steps = x.shape[0]
        h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []

        y_i = base_pose
        for i in range(time_steps):
            rnn_in = torch.cat([h_x, z, y_i, x[i]], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(
        self,
        pose: torch.Tensor,
        future_poses: Optional[torch.Tensor],
        trajectory: torch.Tensor,
        std: float = 1.0,
    ) -> Dict[str, torch.Tensor]:

        if future_poses is not None:
            mu, logvar = self.encode(trajectory, future_poses)
            z = self.reparametrize(mu, logvar) if self.training else mu
        else:
            z = (
                torch.randn(
                    pose.shape[0],
                    self.nz,
                    device=pose.device,
                    dtype=pose.dtype,
                )
                * std
            )
            mu = torch.zeros_like(z)
            logvar = torch.zeros_like(z)

        poses = self.decode(pose, trajectory, z).view(
            (trajectory.shape[0], trajectory.shape[1], self.num_joints, 3)
        )
        return {"predicted_poses": poses, "mu": mu, "logvar": logvar}

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

        output = self(base_pose, None, trajectories, std=std)
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

        future_poses = batch["shift_poses"].transpose(0, 1)
        mask = batch["trajectory_mask"].unsqueeze(dim=0)

        future_poses = future_poses * (1 - mask)
        future_poses = future_poses.contiguous().view(
            (future_poses.shape[0], future_poses.shape[1], -1)
        )[1:]
        output = self(
            base_pose,
            future_poses,
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
        losses["kld"] = criterions["kld"](output["mu"], output["logvar"])

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
