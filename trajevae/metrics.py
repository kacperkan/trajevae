import functools
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from scipy.spatial.distance import pdist

from trajevae.utils.skeleton import Skeleton

TensorOrArray = Union[torch.Tensor, np.ndarray]


def is_tensor(obj: Any) -> bool:
    return isinstance(obj, torch.Tensor)


def _preprocess_tensor(tensor: TensorOrArray, biased: bool) -> torch.Tensor:
    """Biased metrics are used in Video3DPose"""
    seq_len, batch_size, num_joints, coords = tensor.shape

    if biased:
        tensor[:, :, 1:] = tensor[:, :, 1:] - tensor[:, :, :1]
        tensor[:, :, 0] = 0.0
        tensor = tensor.reshape((seq_len, batch_size, num_joints * coords))
    else:
        tensor = tensor[:, :, 1:] - tensor[:, :, :1]
        tensor = tensor.reshape(
            (seq_len, batch_size, (num_joints - 1) * coords)
        )
    return tensor


def _preprocess_tensor_without_flattenting(
    tensor: TensorOrArray, biased: bool
) -> torch.Tensor:
    """Biased metrics are used in Video3DPose"""
    if biased:
        tensor[:, :, 1:] = tensor[:, :, 1:] - tensor[:, :, :1]
        tensor[:, :, 0] = 0.0
    else:
        tensor = tensor[:, :, 1:] - tensor[:, :, :1]
    return tensor


def compute_diversity(
    pred: TensorOrArray, *args, biased: bool = False
) -> float:
    if isinstance(pred, torch.Tensor):
        pred.permute((1, 0, 2, 3))
        if pred.shape[0] == 1:
            return 0.0
        if biased:
            pred[:, :, 1:] = pred[:, :, 1:] - pred[:, :, :1]
            pred[:, :, 0] = 0.0
        else:
            pred = pred[:, :, 1:] - pred[:, :, :1]
        pred = pred[:, :, 1:] - pred[:, :, :1]
        dist = torch.pdist(pred.view(pred.shape[0], -1))
    else:
        if pred.shape[0] == 1:
            return 0.0
        if biased:
            pred[:, :, 1:] = pred[:, :, 1:] - pred[:, :, :1]
            pred[:, :, 0] = 0.0
        else:
            pred = pred[:, :, 1:] - pred[:, :, :1]
        dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_mpjpe(
    pred: TensorOrArray, gt: TensorOrArray, *args, biased: bool = False
) -> float:
    if isinstance(pred, torch.Tensor):
        pred = _preprocess_tensor_without_flattenting(
            pred.detach(), biased=biased
        )
        gt = _preprocess_tensor_without_flattenting(gt, biased=biased)

        diff = pred - gt
        dist = diff.norm(dim=-1).mean()
        return dist
    gt = _preprocess_tensor_without_flattenting(gt[None], biased=biased)
    pred = _preprocess_tensor_without_flattenting(pred, biased=biased)
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=-1).mean(axis=(1, 2))
    return dist.min()


def compute_ade(
    pred: TensorOrArray, gt: TensorOrArray, *args, biased: bool = False
) -> float:
    if isinstance(pred, torch.Tensor):
        pred = _preprocess_tensor(pred.detach(), biased=biased)
        gt = _preprocess_tensor(gt, biased=biased)

        diff = pred - gt
        dist = diff.norm(dim=2).mean(dim=0)
        return dist.mean()
    gt = _preprocess_tensor(gt[None], biased=biased)
    pred = _preprocess_tensor(pred, biased=biased)
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(
    pred: TensorOrArray, gt: TensorOrArray, *args, biased: bool = False
) -> float:
    if isinstance(pred, torch.Tensor):
        pred = _preprocess_tensor(pred.detach(), biased=biased)
        gt = _preprocess_tensor(gt, biased=biased)

        diff = pred - gt
        dist = diff.norm(dim=2)[-1]
        return dist.mean()
    gt = _preprocess_tensor(gt[None], biased=biased)
    pred = _preprocess_tensor(pred, biased=biased)

    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mve(
    pred: TensorOrArray, gt: TensorOrArray, *args, biased: bool = False
) -> float:
    if len(gt.shape) == 3:
        gt = gt[None]
    if is_tensor(pred):
        gt = gt.transpose(0, 1)
        pred = pred.transpose(0, 1)
    pred_vel = pred[:, 1:, 0] - pred[:, :-1, 0]
    gt_vel = gt[:, 1:, 0] - gt[:, :-1, 0]

    if is_tensor(pred):
        return torch.norm(pred_vel - gt_vel, dim=-1).mean()
    return np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()


def compute_mpjve(
    pred: TensorOrArray, gt: TensorOrArray, *args, biased: bool = False
) -> float:
    if isinstance(pred, torch.Tensor):
        pred = _preprocess_tensor_without_flattenting(
            pred.detach(), biased=biased
        )
        gt = _preprocess_tensor_without_flattenting(gt, biased=biased)

        pred = pred[1:] - pred[:-1]
        gt = gt[1:] - gt[:-1]

        diff = pred - gt
        dist = diff.norm(dim=-1).mean(dim=(0, 2))
        return dist.mean()

    gt = _preprocess_tensor_without_flattenting(gt[None], biased=biased)
    pred = _preprocess_tensor_without_flattenting(pred, biased=biased)

    pred = pred[:, 1:] - pred[:, :-1]
    gt = gt[:, 1:] - gt[:, :-1]

    diff = pred - gt
    dist = np.linalg.norm(diff, axis=-1).mean(axis=(1, 2))
    return dist.min()


def compute_bone_rmse(
    pred: TensorOrArray,
    gt: TensorOrArray,
    *args,
    skeleton: Skeleton,
    biased: bool = False,
) -> float:
    if len(gt.shape) == 3:
        gt = gt[None]
    pred_dists_bones = pred[:, :, 1:] - pred[:, :, skeleton.parents()[1:]]
    true_dists_bones = gt[:, :, 1:] - gt[:, :, skeleton.parents()[1:]]
    if is_tensor(pred):
        pred_dists_bones = torch.norm(pred_dists_bones, dim=-1)
        true_dists_bones = torch.norm(true_dists_bones, dim=-1)
    else:
        pred_dists_bones = np.linalg.norm(pred_dists_bones, axis=-1)
        true_dists_bones = np.linalg.norm(true_dists_bones, axis=-1)

    rmse = ((pred_dists_bones - true_dists_bones) ** 2).mean(-1)
    if is_tensor(pred):
        rmse = rmse.sqrt()
    else:
        rmse = np.sqrt(rmse)
    return rmse.mean()


def _compute_multimodal_metric(
    pred: TensorOrArray,
    gt: TensorOrArray,
    gt_multi: List[TensorOrArray],
    metric: Callable,
    biased: bool = False,
    **kwargs,
) -> float:
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = metric(pred, gt_multi_i, biased=biased, **kwargs)
        gt_dist.append(dist)
    res = np.array(gt_dist).mean()
    return res


compute_mmade = functools.partial(
    _compute_multimodal_metric, metric=compute_ade
)
compute_mmfde = functools.partial(
    _compute_multimodal_metric, metric=compute_fde
)
compute_mmmve = functools.partial(
    _compute_multimodal_metric, metric=compute_mve
)
compute_bone_mmrmse = functools.partial(
    _compute_multimodal_metric, metric=compute_bone_rmse
)
compute_mmmpjpe = functools.partial(
    _compute_multimodal_metric, metric=compute_mpjpe
)
compute_mmmpjve = functools.partial(
    _compute_multimodal_metric, metric=compute_mpjve
)


def computer_training_metrics(
    pred: torch.Tensor, gt: torch.Tensor, skeleton: Skeleton
) -> Dict[str, float]:
    name_met = {
        "ade": compute_ade,
        "fde": compute_fde,
        # "mve": compute_mve,
        "mpjpe": compute_mpjpe,
        # "mpjve": compute_mpjve,
        # "bone_rmse": functools.partial(compute_bone_rmse, skeleton=skeleton),
    }
    return {key: metric(pred, gt) for key, metric in name_met.items()}
