import json
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
import wandb
from easydict import EasyDict
from torch import optim
from torch.utils.data import DataLoader
from trajevae.utils import image_utils
from trajevae.utils.general import tensors_to_cuda
from wandb.wandb_run import Run


class BaseModel(nn.Module):
    def __init__(self, config: EasyDict, is_test_run: bool) -> None:
        super().__init__()
        self.config = config
        self.is_test_run = is_test_run

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

        return metrics

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
