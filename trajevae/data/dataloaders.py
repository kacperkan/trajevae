from typing import Optional, Sequence, Set, Tuple, Union

from easydict import EasyDict
from torch.utils.data.dataloader import DataLoader
from trajevae.data.dataset_h36m import DatasetH36MWithLimbTrajectories
from typing_extensions import Literal


def get_dataloaders(
    data_path: str,
    config: EasyDict,
    base_dataset_class_name: Literal["human36m"],
    is_test_run: bool = False,
    is_debug: bool = False,
    actions: Union[Set[str], Literal["all"]] = "all",
    batch_size: Optional[int] = None,
    are_both_valid_loaders: bool = False,
    joint_indices_to_use: Optional[Sequence[int]] = None,
    use_data_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = config.batch_size
    if is_test_run:
        batch_size = 1

    if base_dataset_class_name == "human36m":
        base_dataset_class = DatasetH36MWithLimbTrajectories
    else:
        raise ValueError(
            "Unknown dataset: {}. Available: human36m".format(
                base_dataset_class_name
            )
        )

    train_dataset = base_dataset_class(
        data_path,
        num_steps=(config.num_training_steps * batch_size),
        mode="train",
        t_his=config.t_his,
        t_pred=config.t_pred,
        actions=actions,
        seed=config.seed,
        joint_dropout=config.joint_dropout,
        is_test_run=is_test_run,
        is_valid=are_both_valid_loaders,
        joint_indices_to_use=joint_indices_to_use,
        standardize_data=config.standardize_data,
        use_augmentation=use_data_augmentation,
    )
    valid_dataset = base_dataset_class(
        data_path,
        num_steps=(config.num_training_steps * batch_size),
        mode="test",
        t_his=config.t_his,
        t_pred=config.t_pred,
        actions=actions,
        seed=config.seed,
        joint_dropout=config.joint_dropout,
        is_test_run=is_test_run,
        is_valid=True,
        joint_indices_to_use=joint_indices_to_use,
        scaler=train_dataset.scaler,
        standardize_data=config.standardize_data,
        use_augmentation=use_data_augmentation,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, valid_loader
