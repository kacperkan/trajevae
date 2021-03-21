import concurrent.futures
import csv
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
import trajevae.metrics as trajevae_metrics
from easydict import EasyDict
from functional import seq
from scipy.spatial.distance import pdist, squareform
from torch.utils.data.dataloader import DataLoader
from trajevae.common import (
    copy2cpu,
    create_logger,
    load_config_and_args,
    load_model_module,
)
from trajevae.data.dataloaders import get_dataloaders
from trajevae.models.trajevae import TrajeVAE
from trajevae.utils.general import load_models
from typing_extensions import Literal

logger = create_logger("Score")

stats_func: Dict[str, Callable] = OrderedDict(
    {
        "Diversity": trajevae_metrics.compute_diversity,
        "ADE": trajevae_metrics.compute_ade,
        "MMADE": trajevae_metrics.compute_mmade,
        "FDE": trajevae_metrics.compute_fde,
        "MMFDE": trajevae_metrics.compute_mmfde,
        # "MVE": trajevae_metrics.compute_mve,
        # "MMMVE": trajevae_metrics.compute_mmmve,
        # "Bone-RMSE": functools.partial(
        #     trajevae_metrics.compute_bone_rmse,
        #     skeleton=test_dataset.dataset.skeleton,
        # ),
        # "Bone-MMRMSE": functools.partial(
        #     trajevae_metrics.compute_bone_mmrmse,
        #     skeleton=test_dataset.dataset.skeleton,
        # ),
        "MPJPE": trajevae_metrics.compute_mpjpe,
        # "MMMPJPE": trajevae_metrics.compute_mmmpjpe,
        # "MPJVE": trajevae_metrics.compute_mpjve,
        # "MMMPJVE": trajevae_metrics.compute_mmmpjve,
    }
)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_gt(data: np.ndarray, t_his: int) -> np.ndarray:
    gt = data[:, t_his:, :]
    return gt


def get_prediction(
    data: np.ndarray,
    trajectory: np.ndarray,
    model: TrajeVAE,
    sample_num: int,
    device: torch.device,
    dtype: torch.dtype,
    t_his: int,
    t_pred: int,
    target_seq_len: Optional[int],
    deterministic: bool = False,
    std: float = 1.0,
) -> np.ndarray:
    poses = data.permute(1, 0, 2, 3).contiguous()
    trajectories = torch.tensor(trajectory, device=device, dtype=dtype)

    # future_poses = poses[1:].view((poses.shape[0] - 1, poses.shape[1], -1))
    future_poses = None

    Y = model.sample(
        poses[0],
        future_poses,
        trajectories,
        poses.shape[1],
        target_seq_len or poses.shape[0] - 1,
        device,
        dtype,
        t_pred,
        sample_num,
        deterministic=deterministic,
        std=std,
    )

    return Y


def get_action_key(
    dataset_type: Literal["human36m"], current_action: str
) -> str:
    return current_action.split()[0]


@torch.no_grad()
def compute_stats(
    data_path: str,
    cfg: EasyDict,
    model: TrajeVAE,
    device: torch.device,
    dtype: torch.dtype,
    use_pseudo_past_frames: bool,
    is_test_run: bool,
    metrics_to_use: Sequence[str] = tuple(stats_func.keys()),
):
    original_t_pred = cfg.t_pred
    cfg.t_pred = cfg.target_seq_len or cfg.t_pred
    _, test_dataset = get_dataloaders(
        data_path=data_path,
        config=cfg,
        base_dataset_class_name=cfg.dataset_type,
        is_test_run=is_test_run,
        is_debug=False,
        actions=cfg.actions,
        batch_size=cfg.batch_size,
        joint_indices_to_use=cfg.joint_indices_to_use,
    )
    traj_gt_arr = np.array(get_multimodal_gt(cfg, test_dataset), dtype=object)
    filtered_stats_func = {
        key: func for key, func in stats_func.items() if key in metrics_to_use
    }

    num_samples = 0
    pbar = tqdm.tqdm(test_dataset)
    gt_total = []
    gt_multi_total = []
    pred_total = []
    actions = []
    for batch in pbar:
        data = batch["shift_poses"].to(device)
        trajectory = batch["trajectory"].to(device)
        indices = copy2cpu(batch["index"])
        num_samples += len(batch)
        gt = get_gt(data, cfg.t_his)
        gt_multi = traj_gt_arr[indices]
        t_pred = cfg.t_pred

        pred = get_prediction(
            data=data[:, : original_t_pred + 1],
            trajectory=trajectory[:, :original_t_pred],
            model=model,
            device=device,
            dtype=dtype,
            t_his=cfg.t_his,
            sample_num=cfg.nk,
            t_pred=t_pred,
            target_seq_len=cfg.target_seq_len,
            deterministic=cfg.deterministic,
            std=cfg.sampling_std,
        )
        if use_pseudo_past_frames:
            pred = pred[:, :, cfg.t_his - 1 :]
            gt = gt[:, cfg.t_his - 1 :]
            for i in range(len(gt_multi)):
                gt_multi[i] = gt_multi[i][:, cfg.t_his - 1 :]
        actions.extend(
            seq(batch["action"])
            .map(lambda x: get_action_key(cfg.dataset_type, x))
            .to_list()
        )

        gt_total.append(copy2cpu(gt))
        gt_multi_total.append(gt_multi)
        pred_total.append(copy2cpu(pred))

    pbar.close()

    gt = np.concatenate(gt_total, axis=0)
    gt_multi = np.concatenate(gt_multi_total)
    pred = np.concatenate(pred_total, axis=0)

    stats_names = list(filtered_stats_func.keys())
    stats_meter: Dict[str, Dict[str, AverageMeter]] = defaultdict(
        lambda: defaultdict(AverageMeter)
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(stats_names)
    ) as pool:
        for gt_sample, pred_i, gt_multi_sample, action in tqdm.tqdm(
            zip(gt, pred, gt_multi, actions), total=len(gt)
        ):
            func_name = {
                pool.submit(func, pred_i, gt_sample, gt_multi_sample): name
                for name, func in filtered_stats_func.items()
            }
            for future in concurrent.futures.as_completed(func_name):
                val = future.result()
                stats_meter[action][func_name[future]].update(val)
                stats_meter["average"][func_name[future]].update(val)

    logger.info("=" * 80)
    for action, data in stats_meter.items():
        for name, meter in data.items():
            str_stats = f"{action}: {name} -> {meter.avg:.4f}"
            logger.info(str_stats)
    logger.info("=" * 80)

    if (
        cfg.experiment_subfolder is not None
        and len(cfg.experiment_subfolder) > 0
    ):
        results_dir = (
            Path("outputs")
            / cfg.results_dir
            / cfg.experiment_name
            / cfg.experiment_subfolder
            / "stats_{}.csv".format(cfg.num_seeds)
        )
    else:
        results_dir = (
            Path("outputs")
            / cfg.results_dir
            / cfg.experiment_name
            / "stats_{}.csv".format(cfg.num_seeds)
        )
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(results_dir.as_posix(), "w") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["value", "metric", "action"]
        )
        writer.writeheader()
        for action, data in stats_meter.items():
            for stats, meter in data.items():
                new_meter = {
                    "value": meter.avg,
                    "metric": stats,
                    "action": action,
                }
                writer.writerow(new_meter)


@torch.no_grad()
def compute_multimodal_trajectories(
    data_path: str,
    cfg: EasyDict,
    model: TrajeVAE,
    device: torch.device,
    dtype: torch.dtype,
    is_test_run: bool,
):
    original_t_pred = cfg.t_pred
    cfg.t_pred = cfg.target_seq_len or cfg.t_pred
    _, test_dataset = get_dataloaders(
        data_path=data_path,
        config=cfg,
        base_dataset_class_name=cfg.dataset_type,
        is_test_run=is_test_run,
        is_debug=False,
        actions=cfg.actions,
        batch_size=cfg.batch_size,
        joint_indices_to_use=cfg.joint_indices_to_use,
    )

    shift_poses, trajectories, all_actions = get_possible_trajectories(
        cfg, test_dataset
    )
    batch_size = cfg.batch_size
    data_size = len(shift_poses)

    num_samples = 0
    pbar = tqdm.trange(0, data_size // batch_size + 1, batch_size)
    gt_total = []
    pred_total = []
    actions = []
    for i in pbar:
        data = shift_poses[i : i + batch_size].to(device)
        trajectory = trajectories[i : i + batch_size].to(device)
        num_samples += len(data)
        gt = get_gt(data, cfg.t_his)
        t_pred = cfg.t_pred

        pred = get_prediction(
            data=data[:, : original_t_pred + 1],
            trajectory=trajectory[:, :original_t_pred],
            model=model,
            device=device,
            dtype=dtype,
            t_his=cfg.t_his,
            sample_num=cfg.nk,
            t_pred=t_pred,
            target_seq_len=cfg.target_seq_len,
            deterministic=cfg.deterministic,
            std=cfg.sampling_std,
        )
        actions.extend(
            seq(all_actions[i : i + batch_size])
            .map(lambda x: get_action_key(cfg.dataset_type, x))
            .to_list()
        )

        gt_total.append(copy2cpu(gt))
        pred_total.append(copy2cpu(pred))

    pbar.close()

    gt = np.concatenate(gt_total, axis=0)
    pred = np.concatenate(pred_total, axis=0)

    local_stats_func = {
        name: func
        for name, func in stats_func.items()
        if name not in ["MMADE", "MMFDE", "MPJPE"]
    }

    stats_names = list(local_stats_func.keys())
    stats_meter: Dict[str, Dict[str, AverageMeter]] = defaultdict(
        lambda: defaultdict(AverageMeter)
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(stats_names)
    ) as pool:
        for gt_sample, pred_i, action in tqdm.tqdm(
            zip(gt, pred, actions), total=len(gt)
        ):
            func_name = {
                pool.submit(func, pred_i, gt_sample): name
                for name, func in local_stats_func.items()
            }
            for future in concurrent.futures.as_completed(func_name):
                val = future.result()
                stats_meter[action][func_name[future]].update(val)
                stats_meter["average"][func_name[future]].update(val)

    logger.info("=" * 80)
    for action, data in stats_meter.items():
        for name, meter in data.items():
            str_stats = f"{action}: {name} -> {meter.avg:.4f}"
            logger.info(str_stats)
    logger.info("=" * 80)

    if (
        cfg.experiment_subfolder is not None
        and len(cfg.experiment_subfolder) > 0
    ):
        results_dir = (
            Path("outputs")
            / cfg.results_dir
            / cfg.experiment_name
            / cfg.experiment_subfolder
            / "stats_{}.csv".format(cfg.num_seeds)
        )
    else:
        results_dir = (
            Path("outputs")
            / cfg.results_dir
            / cfg.experiment_name
            / "stats_{}.csv".format(cfg.num_seeds)
        )
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    with open(results_dir.as_posix(), "w") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["value", "metric", "action"]
        )
        writer.writeheader()
        for action, data in stats_meter.items():
            for stats, meter in data.items():
                new_meter = {
                    "value": meter.avg,
                    "metric": stats,
                    "action": action,
                }
                writer.writerow(new_meter)


def get_multimodal_gt(config: EasyDict, data_loader: DataLoader):
    all_data = []
    for data in data_loader:
        samples = data["shift_poses"]
        all_data.append(copy2cpu(samples))
    all_data_np = np.concatenate(all_data, axis=0)
    all_start_pose = all_data_np[:, config.t_his - 1, :]
    pd = squareform(
        pdist(all_start_pose.reshape((all_start_pose.shape[0], -1)))
    )
    traj_gt_arr = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < config.multimodal_threshold)
        traj_gt_arr.append(all_data_np[ind][:, config.t_his :, :])
    return traj_gt_arr


def get_possible_trajectories(
    config: EasyDict, data_loader: DataLoader
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    all_data = []
    all_start_poses = []
    all_trajectories = []
    all_actions = []
    joint_indices_to_use = config.joint_indices_to_use

    for data in data_loader:
        all_start_poses.append(copy2cpu(data["base_pose"]))
        all_trajectories.append(copy2cpu(data["trajectory"]))
        all_data.append(copy2cpu(data["shift_poses"]))
        all_actions.append(np.array(data["action"]))

    all_data_tensor = np.concatenate(all_data, axis=0)
    all_start_poses_tensor = np.concatenate(all_start_poses, axis=0)
    all_trajectories_tensor = np.concatenate(all_trajectories, axis=0)
    all_actions_tensor = np.concatenate(all_actions, axis=0)

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
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < 0.01)

        output_data.append(all_data_tensor[ind])
        output_trajectories.append(
            np.repeat(
                all_trajectories_tensor[[i]], repeats=len(ind[0]), axis=0
            )
        )
        output_actions.append(all_actions_tensor[ind])

    output_data_np = np.concatenate(output_data, axis=0)
    output_trajectories_np = np.concatenate(output_trajectories, axis=0)
    output_actions_np = np.concatenate(output_actions, axis=0)

    return (
        torch.from_numpy(output_data_np),
        torch.from_numpy(output_trajectories_np),
        output_actions_np,
    )


def evaluate():
    config, args = load_config_and_args()
    config.joint_indices_to_use = (
        args["joint_indices_to_use"] or config.joint_indices_to_use
    )

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    torch.set_grad_enabled(False)

    model = load_model_module(
        "trajevae.models.{}".format(args["module_name"])
    )(config, args["test_run"]).to(device)
    model = model.eval()

    model_path = Path("outputs") / "models" / config.experiment_name
    load_models(model_path.as_posix(), ["model"], [model])

    compute_stats(
        data_path=args["data_folder"],
        cfg=config,
        model=model,
        device=device,
        dtype=dtype,
        use_pseudo_past_frames=False,
        is_test_run=args["test_run"],
    )


if __name__ == "__main__":
    evaluate()
