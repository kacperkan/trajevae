import json
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import tqdm
import wandb
from easydict import EasyDict
from torch import optim
from torch.utils.data.dataloader import DataLoader
from wandb.wandb_run import Run

import trajevae.models.layers as trajegan_layers
import trajevae.utils.general as general_utils
from trajevae.metrics import computer_training_metrics
from trajevae.utils import image_utils
from trajevae.utils.general import tensors_to_cuda


class Decoder(nn.Module):
    def __init__(
        self,
        num_joints: int,
        hidden_feats: int,
        num_heads: int,
        transformer_layers: int,
        dropout: float,
        use_norm: bool,
        use_dct: bool,
    ):
        super().__init__()
        self.nh_rnn = hidden_feats
        self.dropout = dropout
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.positional_encoding = trajegan_layers.PositionalEncoding(
            self.nh_rnn * 2, self.dropout, max_len=4096
        )
        self.use_dct = use_dct

        norm_cls = nn.LayerNorm if use_norm else nn.Identity()
        self.post_idct_decoder_mlp = nn.Sequential(
            nn.Linear(self.nh_rnn * 3, self.nh_rnn * 2),
            norm_cls(self.nh_rnn * 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(self.nh_rnn * 2, self.nh_rnn * 2),
        )
        #  poses
        self.d_model = trajegan_layers.create_transformer_encoder_layer(
            self.nh_rnn * 2,
            self.num_heads,
            self.transformer_layers,
            self.dropout,
        )
        self.d_out = nn.Sequential(
            nn.Linear(self.nh_rnn * 2, self.nh_rnn * 2),
            norm_cls(self.nh_rnn * 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(self.nh_rnn * 2, num_joints * 3),
        )

    def forward(
        self,
        dct_features: torch.Tensor,
        initial_pose: torch.Tensor,
        encoded_initial_pose: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        latent = dct_features
        if self.use_dct:
            latent = general_utils.idct(dct_features, dim=0, num=seq_len)
        latent = torch.cat((encoded_initial_pose, latent), dim=-1)
        latent = self.post_idct_decoder_mlp(latent)

        y = self.d_model(self.positional_encoding(latent))
        y = self.d_out(y).cumsum(dim=0) + initial_pose.unsqueeze(dim=0)

        return {
            "predicted_poses": y.view(
                (seq_len, y.shape[1], self.num_joints, 3)
            )
        }


class InitialPoseEncoder(nn.Module):
    def __init__(self, num_joints: int, latent_size: int, use_norm: bool):
        super().__init__()
        self.num_joints = num_joints
        norm_cls = nn.LayerNorm if use_norm else nn.Identity()

        self.layers = nn.Sequential(
            nn.Linear(3 * self.num_joints, latent_size),
            norm_cls(latent_size),
            nn.LeakyReLU(0.1),
            nn.Linear(latent_size, latent_size),
            norm_cls(latent_size),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TrajeVAE(nn.Module):
    def __init__(self, config: EasyDict, is_test_run: bool) -> None:
        super().__init__()

        self.max_len = 4096
        self.transformer_layers = config.transformer_layers
        self.num_heads = config.attn_heads
        self.encoder_dropout = config.encoder_dropout
        self.decoder_dropout = config.decoder_dropout
        self.use_norm = not is_test_run
        self.use_vae = config.use_vae
        self.trajectory_resolution = config.num_joints
        self.feat_multiplier = config.features_multiplier
        self.learnable_initial_pose = config.learnable_initial_pose
        self.num_joints = config.num_joints
        self.dct_components = config.dct_components
        self.joint_dropout = config.joint_dropout
        self.mask_future_poses = config.mask_future_poses
        self.use_dct = config.use_dct
        self.use_learnable_prior = config.use_learnable_prior

        self.latent_size = 256

        self.positional_encoding = trajegan_layers.PositionalEncoding(
            self.latent_size * 2, self.encoder_dropout, max_len=self.max_len
        )

        # encode
        self.initial_pose_encoder = InitialPoseEncoder(
            self.num_joints, self.latent_size, self.use_norm
        )
        self.sequence_pose_encoder = InitialPoseEncoder(
            self.num_joints, self.latent_size, self.use_norm
        )

        self.pre_dct_x_model = (
            trajegan_layers.create_transformer_encoder_layer(
                self.latent_size * 2,
                self.num_heads,
                self.transformer_layers,
                self.encoder_dropout,
            )
        )
        self.post_dct_x_model_1 = (
            trajegan_layers.create_transformer_encoder_layer(
                self.latent_size * 2,
                self.num_heads,
                self.transformer_layers,
                self.encoder_dropout,
            )
        )

        if self.use_learnable_prior:
            self.post_dct_x_model_2_pose = (
                trajegan_layers.create_transformer_encoder_layer(
                    self.latent_size * 2,
                    self.num_heads,
                    self.transformer_layers,
                    self.encoder_dropout,
                )
            )
        self.post_dct_x_model_2_traj = (
            trajegan_layers.create_transformer_encoder_layer(
                self.latent_size * 2,
                self.num_heads,
                self.transformer_layers,
                self.encoder_dropout,
            )
        )

        if self.use_learnable_prior:
            self.x_mu_pose = nn.Linear(
                self.latent_size * 2, self.latent_size * 2
            )
            self.x_logvar_pose = nn.Linear(
                self.latent_size * 2, self.latent_size * 2
            )

        self.x_mu_traj = nn.Linear(self.latent_size * 2, self.latent_size * 2)
        self.x_logvar_traj = nn.Linear(
            self.latent_size * 2, self.latent_size * 2
        )

        self.decoder = Decoder(
            self.num_joints,
            self.latent_size,
            self.num_heads,
            self.transformer_layers,
            self.decoder_dropout,
            self.use_norm,
            self.use_dct,
        )

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

    @property
    def num_parameters(self) -> int:
        total_params = 0
        for param in self.parameters():
            total_params += np.prod(param.shape)
        return total_params

    @property
    def num_trainable_parameters(self) -> int:
        total_params = 0
        for param in self.parameters():
            if param.requires_grad:
                total_params += np.prod(param.shape)
        return total_params

    @property
    def num_non_trainable_parameters(self) -> int:
        total_params = 0
        for param in self.parameters():
            if not param.requires_grad:
                total_params += np.prod(param.shape)
        return total_params

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        noise = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return noise * std + mu

    def encode(
        self,
        x: torch.Tensor,
        encoding_module: nn.Module,
        mu_pred: nn.Module,
        logvar_pred: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        before_dct = x = self.pre_dct_x_model(self.positional_encoding(x))
        if self.use_dct:
            x = general_utils.dct(x, dim=0)
        after_dct = x = self.post_dct_x_model_1(x)
        x = encoding_module(x)
        mu, logvar = mu_pred(x), logvar_pred(x)
        z = self.reparameterize(mu, logvar)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "after_dct": after_dct,
            "before_dct": before_dct,
        }

    def forward(
        self,
        pose: torch.Tensor,
        future_poses: Optional[torch.Tensor],
        trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        initial_pose_emb = self.initial_pose_encoder(pose)
        seq_initial_pose_emb = initial_pose_emb.unsqueeze(dim=0).repeat(
            (trajectory.shape[0], 1, 1)
        )
        data = torch.cat(
            (self.sequence_pose_encoder(trajectory), seq_initial_pose_emb),
            dim=-1,
        )
        trajectory_out = self.encode(
            data,
            self.post_dct_x_model_2_traj,
            self.x_mu_traj,
            self.x_logvar_traj,
        )
        out = {"traj_" + key: val for key, val in trajectory_out.items()}

        if future_poses is not None and self.use_learnable_prior:
            data = torch.cat(
                (
                    self.sequence_pose_encoder(future_poses),
                    seq_initial_pose_emb,
                ),
                dim=-1,
            )
            pose_out = self.encode(
                data,
                self.post_dct_x_model_2_pose,
                self.x_mu_pose,
                self.x_logvar_pose,
            )
            pose_decoded = self.decoder(
                pose_out["z"],
                pose,
                seq_initial_pose_emb,
                future_poses.shape[0],
            )
            out.update({"pose_" + key: val for key, val in pose_out.items()})
            out.update(pose_decoded)
        else:
            trajectory_decoded = self.decoder(
                trajectory_out["z"],
                pose,
                seq_initial_pose_emb,
                trajectory.shape[0],
            )
            out.update(trajectory_decoded)

        return out

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
        initial_pose_emb = self.initial_pose_encoder(base_pose)
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
            .transpose(0, 1)
        )
        data = torch.cat(
            (
                self.sequence_pose_encoder(trajectories),
                initial_pose_emb.unsqueeze(dim=0).repeat(
                    (trajectories.shape[0], 1, 1)
                ),
            ),
            dim=-1,
        )
        trajectory_out = self.encode(
            data,
            self.post_dct_x_model_2_traj,
            self.x_mu_traj,
            self.x_logvar_traj,
        )

        if deterministic:
            output = self.decoder(
                trajectory_out["mu"],
                base_pose,
                initial_pose_emb.unsqueeze(dim=0).repeat((seq_len, 1, 1)),
                seq_len,
            )
        else:
            z = (
                trajectory_out["z"] - trajectory_out["mu"]
            ) * std + trajectory_out["mu"]
            output = self.decoder(
                z,
                base_pose,
                initial_pose_emb.unsqueeze(dim=0).repeat((seq_len, 1, 1)),
                seq_len,
            )
        output_poses = output["predicted_poses"]
        output_poses = output_poses.view(
            (seq_len, batch_size, num, self.num_joints, -1)
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

        if self.mask_future_poses:
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

        real_poses = batch["shift_poses"][:, 1:].transpose(0, 1)
        fake_poses = output["predicted_poses"]

        losses["joint_rec"] = criterions["joint_rec"](
            fake_poses * (1 - mask), real_poses * (1 - mask)
        )
        losses["traj_rec"] = criterions["traj_rec"](
            fake_poses * mask, real_poses * mask
        )

        if self.use_learnable_prior:
            losses["ft2traj_kld"] = criterions["kld"](
                output["pose_mu"],
                output["pose_logvar"],
                output["traj_mu"],
                output["traj_logvar"],
            )
        else:
            losses["ft2traj_kld"] = criterions["kld"](
                output["traj_mu"], output["traj_logvar"]
            )

        loss: torch.Tensor = sum(losses.values())

        out_metrics = {
            "generator/{}".format(key): value.item()
            for key, value in losses.items()
        }
        out_metrics["generator/total"] = loss.item()
        out_metrics.update(
            computer_training_metrics(
                # calculating without the root node
                output["predicted_poses"],
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
                    output["predicted_poses"].transpose(0, 1),
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
            base_pose = last_batch["base_pose"]
            base_pose = base_pose.view((base_pose.shape[0], -1))

            output = self(
                base_pose,
                None,
                last_batch["trajectory"]
                .view(
                    (
                        last_batch["trajectory"].shape[0],
                        last_batch["trajectory"].shape[1],
                        -1,
                    )
                )
                .transpose(0, 1),
            )
            fake_poses = output["predicted_poses"]
            real_poses = last_batch["shift_poses"][:, 1:].transpose(0, 1)
            self.log_images(
                writer,
                real_poses.transpose(0, 1),
                fake_poses.transpose(0, 1),
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
