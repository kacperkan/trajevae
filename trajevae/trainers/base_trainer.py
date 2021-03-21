import importlib
import pickle
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import tqdm
import trajevae.models.losses as trajegan_losses
import wandb
from easydict import EasyDict
from trajevae.common import drop_config_to, load_model_module
from trajevae.data.dataloaders import get_dataloaders
from trajevae.utils.general import (
    enable_grads,
    load_models,
    save_models,
    tensors_to_cuda,
)
from wandb.wandb_run import Run

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True


def get_criterions(config: Dict[str, Any]) -> Dict[str, nn.Module]:
    output = {}
    for crit_name, crit_config in config.items():
        output[crit_name] = getattr(trajegan_losses, crit_config["class"])(
            **crit_config["params"]
        )
    return output


def train(
    model: nn.Module,
    data_path: str,
    output_model_folder: str,
    config: EasyDict,
    load_pretrained_model: bool,
    data_type: str,
    is_debug: bool,
    is_test_run: bool,
):
    use_cuda = torch.cuda.is_available()

    output_model_path = Path("outputs") / "models" / output_model_folder

    output_model_path.mkdir(exist_ok=True, parents=True)
    drop_config_to(config, output_model_path)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if use_cuda:
        model = model.cuda()

    optimizer_generator = optim.Adam(
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        params=model.parameters(),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer_generator,
        step_size=config.scheduler_step,
        gamma=config.scheduler_gamma,
    )

    load_pretrained = load_pretrained_model and output_model_path.exists()
    if load_pretrained:
        load_models(
            output_model_path.as_posix(),
            ["model", "optimizer_generator", "scheduler"],
            [model, optimizer_generator, scheduler],
        )

    run: Optional[Run] = None
    if not is_debug:
        prefix = "[test]" if is_test_run else ""
        if config.dataset_type == "human36m":
            group_name = "trajevae"
        else:
            group_name = "trajevae-{}".format(config.dataset_type)
        run = wandb.init(
            project="trajegan".format(),
            reinit=not load_pretrained,
            group=group_name,
            job_type="train",
            name=prefix + output_model_folder.replace("trajevae", ""),
        )
        if run is not None:
            run.config.update(config)
            run.watch(model)

    global_step = 0
    if load_pretrained and (output_model_path / "metadata.pkl").exists():
        with open(output_model_path / "metadata.pkl", "rb") as f:
            global_step = pickle.load(f)

    train_loader, valid_loader = get_dataloaders(
        data_path=data_path,
        config=config,
        base_dataset_class_name=config.dataset_type,
        is_test_run=is_test_run,
        is_debug=is_debug,
        actions="all",
        joint_indices_to_use=None,
    )
    gen_pbar = tqdm.tqdm(total=len(train_loader), position=0)
    gen_pbar.update(global_step)

    criterions = get_criterions(config.criterions)

    best_mpjpe = np.inf

    model = model.train()
    for batch in train_loader:
        if global_step >= config.num_training_steps:
            break
        if use_cuda:
            batch = tensors_to_cuda(batch, torch.device("cuda"))
        if is_debug:
            batch = enable_grads(batch)

        # step
        metrics_dict = model.train_step(
            # optimizer
            optimizer_generator,
            # scheduler
            scheduler,
            # initial data
            batch,
            # configs
            config,
            # additional data
            {
                "skeleton": train_loader.dataset.skeleton,
                "metadata": train_loader.dataset.metadata,
                "visualize_every_nth_pose": (config.visualize_every_nth_pose),
                "scaler": train_loader.dataset.scaler,
            },
            # criterions
            criterions,
            run,
            global_step,
        )

        gen_pbar.set_postfix(
            OrderedDict(
                {key: "%.4f" % value for key, value in metrics_dict.items()}
            )
        )

        global_step += 1
        gen_pbar.update(1)

        if global_step % config.validation_frequency == 0:
            model = model.eval()
            with torch.no_grad():
                valid_metrics = model.validate(
                    valid_loader,
                    config,
                    {
                        "skeleton": train_loader.dataset.skeleton,
                        "metadata": train_loader.dataset.metadata,
                        "visualize_every_nth_pose": (
                            config.visualize_every_nth_pose
                        ),
                        "scaler": train_loader.dataset.scaler,
                    },
                    criterions,
                    run,
                    global_step,
                )

            model = model.train()

            if valid_metrics["mpjpe"] < best_mpjpe:
                best_mpjpe = valid_metrics["mpjpe"]
                save_models(
                    output_model_path.as_posix(),
                    ["model", "optimizer_generator", "scheduler"],
                    [model, optimizer_generator, scheduler],
                )

                with open(output_model_path / "metadata.pkl", "wb") as f:
                    pickle.dump(
                        global_step,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                print(
                    "Saved model at {} with MPJPE equal to: {:.4f}".format(
                        global_step, best_mpjpe
                    )
                )

    gen_pbar.close()
    if run is not None:
        run.finish()


def main():
    from trajevae.common import get_args

    args = get_args()
    config = getattr(
        importlib.import_module("configs.{}".format(args["config_name"])),
        "CONFIG",
    )

    if args["disable_masking_future_poses"]:
        config.mask_future_poses = False

    if args["disable_dct"]:
        config.use_dct = False

    if args["disable_data_augmentation"]:
        config.use_data_augmentation = False

    if args["disable_learnable_prior"]:
        config.use_learnable_prior = False
        if "kld" in config.criterions:
            config.criterions["kld"]["class"] = "KLD"

    if len(args["suffix"]) > 0:
        config.experiment_name += "-{}".format(args["suffix"])

    model = load_model_module(
        "trajevae.models.{}".format(args["module_name"])
    )(config, args["test_run"])
    train(
        model,
        args["data_folder"],
        config.experiment_name,
        config,
        config.load_pretrained,
        config.dataset_type,
        args["debug"],
        args["test_run"],
    )


if __name__ == "__main__":
    main()
