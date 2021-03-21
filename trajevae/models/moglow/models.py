import copy
import json
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
import wandb
from easydict import EasyDict
from torch import optim
from torch.utils.data import DataLoader
from trajevae.metrics import computer_training_metrics
from trajevae.utils import image_utils
from trajevae.utils.general import tensors_to_cuda
from wandb.wandb_run import Run

from . import modules, thops


def nan_throw(tensor, name="tensor"):
    stop = False
    if (tensor != tensor).any():
        print(name + " has nans")
        stop = True
    if torch.isinf(tensor).any():
        print(name + " has infs")
        stop = True
    if stop:
        print(name + ": " + str(tensor))
        # raise ValueError(name + ' contains nans of infs')


def f(
    in_channels,
    out_channels,
    hidden_channels,
    cond_channels,
    network_model,
    num_layers,
):
    if network_model == "LSTM":
        return modules.LSTM(
            in_channels + cond_channels,
            hidden_channels,
            out_channels,
            num_layers,
        )
    if network_model == "GRU":
        return modules.GRU(
            in_channels + cond_channels,
            hidden_channels,
            out_channels,
            num_layers,
        )
    if network_model == "FF":
        return nn.Sequential(
            nn.Linear(in_channels + cond_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(inplace=False),
            modules.LinearZeroInit(hidden_channels, out_channels),
        )


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    NetworkModel = ["LSTM", "GRU", "FF"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        cond_channels,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        network_model="LSTM",
        num_layers=2,
        LU_decomposed=False,
    ):

        # check configures
        assert (
            flow_coupling in FlowStep.FlowCoupling
        ), "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert (
            network_model in FlowStep.NetworkModel
        ), "network_model should be in `{}`".format(FlowStep.NetworkModel)
        assert (
            flow_permutation in FlowStep.FlowPermutation
        ), "float_permutation should be in `{}`".format(
            FlowStep.FlowPermutation.keys()
        )
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed
            )
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(
                in_channels // 2,
                in_channels - in_channels // 2,
                hidden_channels,
                cond_channels,
                network_model,
                num_layers,
            )
        elif flow_coupling == "affine":
            print("affine: in_channels = " + str(in_channels))
            self.f = f(
                in_channels // 2,
                2 * (in_channels - in_channels // 2),
                hidden_channels,
                cond_channels,
                network_model,
                num_layers,
            )
            print("Flowstep affine layer: " + str(in_channels))

    def init_lstm_hidden(self):
        if self.network_model == "LSTM" or self.network_model == "GRU":
            self.f.init_hidden()

    def forward(self, input, cond, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, cond, logdet)
        else:
            return self.reverse_flow(input, cond, logdet)

    def normal_flow(self, input, cond, logdet):

        # assert input.size(1) % 2 == 0
        # 1. actnorm
        # z=input
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False
        )
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")
        z1_cond = torch.cat((z1, cond), dim=1)
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2]) + logdet

        z = thops.cat_feature(z1, z2)
        return z, cond, logdet

    def reverse_flow(self, input, cond, logdet):
        # 1.coupling
        z1, z2 = thops.split_feature(input, "split")
        z1_cond = torch.cat((z1, cond), dim=1)

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = thops.split_feature(h, "cross")
            nan_throw(shift, "shift")
            nan_throw(scale, "scale")
            nan_throw(z2, "z2 unscaled")
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2]) + logdet

        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True
        )
        nan_throw(z, "z permute_" + str(self.flow_permutation))
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, cond, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        x_channels,
        hidden_channels,
        cond_channels,
        K,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        network_model="LSTM",
        num_layers=2,
        LU_decomposed=False,
    ):

        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        N = cond_channels
        for _ in range(K):
            self.layers.append(
                FlowStep(
                    in_channels=x_channels,
                    hidden_channels=hidden_channels,
                    cond_channels=N,
                    actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation,
                    flow_coupling=flow_coupling,
                    network_model=network_model,
                    num_layers=2,
                    LU_decomposed=LU_decomposed,
                )
            )
            self.output_shapes.append([-1, x_channels, 1])

    def init_lstm_hidden(self):
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                layer.init_lstm_hidden()

    def forward(self, z, cond, logdet=0.0, reverse=False, eps_std=None):
        if not reverse:
            for layer in self.layers:
                z, cond, logdet = layer(z, cond, logdet, reverse=False)
            return z, logdet
        else:
            for i, layer in enumerate(reversed(self.layers)):
                z, cond, logdet = layer(z, cond, logdet=0, reverse=True)
            return z


class Glow(nn.Module):
    def __init__(self, config: EasyDict, is_test_run: bool):
        super().__init__()
        self.num_joints = config.num_joints
        self.is_test_run = is_test_run
        self.x_channels = self.num_joints * 3
        self.seqlen = config.seqlen
        self.cond_channels = (
            self.seqlen * self.num_joints * 3 * 2 + self.num_joints * 3
        )
        self.standardize_data = config.standardize_data

        self.flow = FlowNet(
            x_channels=self.x_channels,
            hidden_channels=config.hidden_channels,
            cond_channels=self.cond_channels,
            K=config.K,
            actnorm_scale=config.actnorm_scale,
            flow_permutation=config.flow_permutation,
            flow_coupling=config.flow_coupling,
            network_model=config.network_model,
            num_layers=config.num_layers,
            LU_decomposed=config.LU_decomposed,
        )
        self.config = copy.deepcopy(config)
        self.frame_dropout = config.frame_dropout

        # register prior hidden
        self.z_shape = [self.x_channels, 1]
        self.distribution: Union[modules.GaussianDiag, modules.StudentT]
        if config.distribution == "normal":
            self.distribution = modules.GaussianDiag()
        elif config.distribution == "studentT":
            self.distribution = modules.StudentT(
                config.distribution_param, self.x_channels
            )

    def init_lstm_hidden(self):
        self.flow.init_lstm_hidden()

    def forward(
        self,
        x=None,
        cond=None,
        z=None,
        eps_std=1.0,
        reverse=False,
    ):
        if not reverse:
            return self.normal_flow(x, cond)
        else:
            return self.reverse_flow(z, cond, eps_std)

    def generate_sample(
        self,
        base_pose: torch.Tensor,
        trajectory: torch.Tensor,
        eps_std: float = 1.0,
    ) -> torch.Tensor:
        nn, _, n_features = trajectory.shape
        sampled_all = torch.zeros(
            (trajectory.shape[0], trajectory.shape[1], trajectory.shape[2]),
            dtype=trajectory.dtype,
            device=trajectory.device,
        )
        autoreg = torch.zeros(
            (nn, self.seqlen, n_features),
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        autoreg[:, -1] = base_pose
        self.init_lstm_hidden()

        trajectory = torch.cat(
            (
                trajectory[:, [0]].repeat_interleave(
                    dim=1, repeats=self.seqlen
                ),
                trajectory,
            ),
            dim=1,
        )

        last_pose = base_pose

        for i in range(0, trajectory.shape[1] - self.seqlen):
            control = trajectory[:, i : (i + self.seqlen + 1)]
            cond = torch.cat(
                (autoreg.view((nn, -1)), control.view((nn, -1))), dim=-1
            ).unsqueeze(dim=-1)

            sampled = self(z=None, cond=cond, eps_std=eps_std, reverse=True)
            new_pose = sampled["output"][..., 0] + last_pose
            sampled_all[:, i, :] = new_pose
            autoreg = torch.cat(
                (autoreg[:, 1:], new_pose.unsqueeze(dim=1)), dim=1
            )
            last_pose = new_pose

        return sampled_all

    def normal_flow(self, x, cond):
        n_timesteps = thops.timesteps(x)

        logdet = torch.zeros_like(x[:, 0, 0])

        # encode
        z, objective = self.flow(x, cond, logdet=logdet, reverse=False)

        # prior
        objective += self.distribution.logp(z)

        # return
        nll = (-objective) / float(np.log(2.0) * n_timesteps)
        return {"output": z, "nll": nll}

    def reverse_flow(self, z, cond, eps_std):
        with torch.no_grad():

            z_shape = [cond.shape[0]] + self.z_shape
            if z is None:
                z = self.distribution.sample(
                    z_shape, eps_std, device=cond.device
                )

            x = self.flow(z, cond, eps_std=eps_std, reverse=True)
        return {"output": x}

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    def sample_trajectories(
        self,
        num: int,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        trajectories = torch.zeros(
            (time_steps, batch_size * num, self.latent_size),
            device=device,
            dtype=dtype,
        )
        return trajectories

    def sample(
        self,
        base_pose: torch.Tensor,
        future_poses: Optional[torch.Tensor],
        trajectories: Optional[torch.Tensor],
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
        if trajectories is None:
            trajectories = torch.zeros(
                (base_pose.shape[0] * num, seq_len, self.num_joints * 3)
            )

        trajectories = (
            trajectories.unsqueeze(dim=1)
            .repeat((1, num, 1, 1, 1))
            .view(
                (
                    batch_size * num,
                    trajectories.shape[1],
                    self.num_joints * 3,
                )
            )
        )
        sampled = self.generate_sample(
            base_pose,
            trajectories,
            eps_std=std,
        )
        output_poses = sampled.view(
            (batch_size, num, sampled.shape[1], self.num_joints, 3)
        )

        return output_poses

    @torch.no_grad()
    def concat_sequence(self, seqlen: int, data: torch.Tensor):
        nn, n_timesteps, n_feats = data.shape
        L = n_timesteps - (seqlen - 1)
        inds = torch.zeros((L, seqlen), dtype=torch.long, device=data.device)

        # create indices for the sequences we want
        rng = torch.arange(
            0, n_timesteps, dtype=torch.long, device=data.device
        )
        for ii in range(0, seqlen):
            inds[:, ii] = rng[ii : (n_timesteps - (seqlen - ii - 1))].t()

        # slice each sample into L sequences and store as new samples
        cc = data[:, inds, :]

        # print ("cc: " + str(cc.shape))

        # reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen * n_feats))
        return dd

    @torch.no_grad()
    def mask_frames(self, poses: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = (
                torch.empty(
                    (poses.shape[0], poses.shape[1]),
                    dtype=poses.dtype,
                    device=poses.device,
                )
                .bernoulli_(1 - self.frame_dropout)
                .unsqueeze(dim=-1)
                .repeat((1, 1, poses.shape[-1]))
            )
            return mask * poses
        return poses

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _convert_to_digestable_format(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_poses = batch["shift_poses"].flatten(-2, -1)
        shift_velocities = shift_poses[:, 1:] - shift_poses[:, :-1]
        shift_poses = shift_poses[:, 1:]
        trajectory = batch["trajectory"].view(
            (batch["trajectory"].shape[0], batch["trajectory"].shape[1], -1)
        )
        new_x = self.concat_sequence(1, shift_velocities[:, self.seqlen :])
        autoreg = self.concat_sequence(
            self.seqlen, self.mask_frames(shift_poses[:, :-1])
        )

        control = self.concat_sequence(self.seqlen + 1, trajectory)
        cond = torch.cat((autoreg, control), dim=-1)
        return new_x.transpose(1, 2), cond.transpose(1, 2)

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

        x, cond = self._convert_to_digestable_format(batch)
        self.init_lstm_hidden()

        if global_step == 0:
            self(x, cond if cond is not None else None)
            self.init_lstm_hidden()

        output = self(x=x, cond=cond)

        losses["nll"] = output["nll"].mean()
        loss: torch.Tensor = sum(losses.values())

        out_metrics = {
            "generator/{}".format(key): value.item()
            for key, value in losses.items()
        }
        out_metrics["generator/total"] = loss.item()

        if is_valid:
            real_poses = batch["shift_poses"][:, 1:]
            output_pose = self.generate_sample(
                base_pose,
                batch["trajectory"].view(
                    (
                        batch["trajectory"].shape[0],
                        batch["trajectory"].shape[1],
                        -1,
                    )
                ),
                eps_std=1.0,
            )
            output_pose = output_pose.view(
                (
                    output_pose.shape[0],
                    output_pose.shape[1],
                    self.num_joints,
                    3,
                )
            )

            out_metrics.update(
                computer_training_metrics(
                    output_pose.transpose(0, 1),
                    real_poses.transpose(0, 1),
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
                output_pose = self.generate_sample(
                    base_pose,
                    batch["trajectory"].view(
                        (
                            batch["trajectory"].shape[0],
                            batch["trajectory"].shape[1],
                            -1,
                        )
                    ),
                    eps_std=1.0,
                )
                output_pose = output_pose.view(
                    (
                        output_pose.shape[0],
                        output_pose.shape[1],
                        self.num_joints,
                        3,
                    )
                )
                real_poses = batch["shift_poses"][:, 1:]

                self.log_images(
                    writer,
                    real_poses.transpose(0, 1),
                    output_pose,
                    batch["trajectory"],
                    additional_metadata,
                    global_step,
                    is_valid,
                )
        return loss, output, out_metrics

    def train_step(
        self,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        batch: Dict[str, torch.Tensor],
        config: EasyDict,
        additional_metadata: Dict[str, Any],
        criterions: Dict[str, nn.Module],
        writer: Optional[Run],
        global_step: int,
    ) -> Dict[str, float]:
        optimizer.zero_grad()
        loss_generator, _, generator_to_print = self._generator_step(
            batch,
            config,
            additional_metadata,
            criterions,
            writer,
            global_step,
            is_valid=False,
        )
        loss_generator.backward()
        # if (
        #     self.config.max_grad_clip is not None
        #     and self.config.max_grad_clip > 0
        # ):
        #     torch.nn.utils.clip_grad_value_(
        #         self.parameters(), self.config.max_grad_clip
        #     )
        # if (
        #     self.config.max_grad_norm is not None
        #     and self.config.max_grad_norm > 0
        # ):
        #     torch.nn.utils.clip_grad_norm_(
        #         self.parameters(), self.config.max_grad_norm
        #     )
        optimizer.step()
        to_print = generator_to_print
        if scheduler is not None:
            scheduler.step()
        if writer is not None:
            writer.log(
                {"lr": optimizer.param_groups[0]["lr"]}, step=global_step
            )

        return to_print

    def validate(
        self,
        data_loader: DataLoader,
        config: EasyDict,
        additional_metadata: Dict[str, Any],
        criterions: Dict[str, nn.Module],
        writer: Optional[Run],
        global_step: int,
    ) -> Dict[str, float]:
        metrics_agg: Dict[str, float] = defaultdict(float)

        total_batches = 0
        last_batch: Optional[Dict[str, torch.Tensor]] = None

        for batch in tqdm.tqdm(data_loader, leave=False):
            batch = tensors_to_cuda(batch, self.device)

            _, _, generator_metrics = self._generator_step(
                batch,
                config,
                additional_metadata,
                criterions,
                None,
                global_step,
                is_valid=True,
            )

            for metric_name, metric_value in generator_metrics.items():
                metrics_agg[metric_name] += metric_value
            total_batches += 1
            last_batch = batch

        metrics: Dict[str, Union[torch.Tensor, float]] = {
            metric_name: metric_value / total_batches
            for metric_name, metric_value in metrics_agg.items()
        }
        print(
            json.dumps(
                {
                    key: val if isinstance(val, float) else val.item()
                    for key, val in metrics.items()
                },
                indent=2,
            )
        )

        if writer is not None:
            writer.log(
                {"valid/" + key: valid for key, valid in metrics.items()},
                step=global_step,
            )

        if writer is not None and last_batch is not None:
            real_poses = last_batch["shift_poses"][:, 1:]

            base_pose = last_batch["base_pose"]
            base_pose = base_pose.view((base_pose.shape[0], -1))
            fake_poses = self.generate_sample(
                base_pose,
                batch["trajectory"].view(
                    (
                        batch["trajectory"].shape[0],
                        batch["trajectory"].shape[1],
                        -1,
                    )
                ),
                eps_std=1.0,
            )
            fake_poses = fake_poses.view(
                (
                    fake_poses.shape[0],
                    fake_poses.shape[1],
                    self.num_joints,
                    3,
                )
            )

            self.log_images(
                writer,
                real_poses,
                fake_poses,
                last_batch["trajectory"],
                additional_metadata,
                global_step,
                True,
            )

        del last_batch

        return {
            key: val if isinstance(val, float) else val.item()
            for key, val in metrics.items()
        }

    def log_images(
        self,
        writer: Run,
        real_poses: torch.Tensor,
        fake_poses: torch.Tensor,
        trajectories: torch.Tensor,
        additional_metadata: Dict[str, Any],
        global_step: int,
        is_valid: bool,
    ):
        limit = 4
        prefix = "valid/" if is_valid else "train/"
        fake_poses = fake_poses[:limit]
        real_poses = real_poses[:limit]
        trajectories = trajectories[:limit]

        last_fake_stickman = image_utils.visualize_poses(
            fake_poses,
            additional_metadata["visualize_every_nth_pose"],
            additional_metadata["skeleton"],
            additional_metadata["metadata"],
            trajectories,
        )

        last_real_stickman = image_utils.visualize_poses(
            real_poses,
            additional_metadata["visualize_every_nth_pose"],
            additional_metadata["skeleton"],
            additional_metadata["metadata"],
            trajectories,
        )

        data_to_log = {
            prefix
            + "generated/stickman": wandb.Image(
                torchvision.utils.make_grid(last_fake_stickman)
            ),
            prefix
            + "real/stickman": wandb.Image(
                torchvision.utils.make_grid(last_real_stickman)
            ),
        }

        writer.log(
            data_to_log,
            step=global_step,
        )
