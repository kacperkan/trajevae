import hashlib
import itertools
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch_dct
from torch.utils.tensorboard.writer import SummaryWriter


def save_models(
    directory: str, names: Sequence[str], modules: Sequence[nn.Module]
):
    directory_path = Path(directory)
    for name, module in zip(names, modules):
        torch.save(
            module.state_dict(),
            (directory_path / name).with_suffix(".pt").as_posix(),
        )


def load_models(
    directory: str, names: Sequence[str], modules: Sequence[nn.Module]
):
    directory_path = Path(directory)
    for name, module in zip(names, modules):
        path = (directory_path / name).with_suffix(".pt")
        if path.exists():
            module.load_state_dict(torch.load(path.as_posix()))
        else:
            print(
                "Model {} does not exist and won't be loaded".format(
                    path.as_posix()
                )
            )


def tensors_to_cuda(
    dict_of_tensors: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    out = {}
    for key, val in dict_of_tensors.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def enable_grads(
    dict_of_tensors: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    out = {}
    for key, val in dict_of_tensors.items():
        if isinstance(val, torch.Tensor) and val.dtype in [
            torch.float32,
            torch.float64,
        ]:
            out[key] = val.requires_grad_(True)
        else:
            out[key] = val
    return out


def get_git_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_branch_name() -> str:
    return (
        subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_writer(experiment_name: str, name: str) -> SummaryWriter:
    directory = Path(__file__).parent.parent.parent / "tensorboards"
    directory.mkdir(exist_ok=True, parents=True)

    output_directory = (
        directory
        / "{}-{}-{}".format(
            experiment_name, get_branch_name(), get_git_hash()
        ).strip()
        / name
    )
    if output_directory.exists():
        shutil.rmtree(output_directory.as_posix())
    output_directory.mkdir(parents=True)

    return SummaryWriter(output_directory.as_posix())


def dct(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor = tensor.transpose(dim, -1)
    result = torch_dct.dct(tensor, norm="ortho")
    return result.transpose(dim, -1).contiguous()


def idct(
    tensor: torch.Tensor, dim: int, num: Optional[int] = None
) -> torch.Tensor:
    tensor = tensor.transpose(dim, -1)
    if num is not None:
        if num > tensor.shape[-1]:
            padding = torch.zeros(
                tuple(tensor.shape[:-1]) + (num - tensor.shape[-1],),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            tensor = torch.cat((tensor, padding), dim=-1)
        elif num < tensor.shape[-1]:
            tensor = tensor[..., :num]
    result = torch_dct.idct(tensor, norm="ortho")
    return result.transpose(dim, -1).contiguous()


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def powerset(iterable: Iterable[Any]) -> List[Any]:
    s = list(iterable)
    return list(
        itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    )
